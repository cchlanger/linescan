from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from skimage import io
import pandas as pd
from lmfit import models
from lmfit import Model
from .vis_tools import measure_line_values, read_roi


def linescan(
    image_path,
    roi_path,
    channels,
    number_of_channels,   # passed through to measure_line_values (layout/indexing handled downstream)
    align_channel,
    measure_channel,
    line_width=5,
    normalize=True,
    scaling=0.03525845591290619,
    align=True,
    peak_method="gaussian",   # default now Gaussian
    align_method="poly",      # "poly" (default) or "sigmoid" for half-max offset and overlay
):
    """
    Perform linescan analysis on images using ROI line segments.

    For each ROI line segment, this function:
    1) extracts the line profile from the align_channel and computes an alignment offset as the
       first half-maximum crossing on a smoothed curve:
       - If align_method == "poly": fit a degree-10 polynomial, upsample, then find the first 0.5 crossing.
       - If align_method == "sigmoid": fit a sigmoid (via lmfit), evaluate densely, then find the first 0.5 crossing.
       The same smoothed curve (polynomial or sigmoid) is overlaid on the align channel plot.
    2) extracts the line profile from the measure_channel and estimates the peak position:
       - If peak_method == "gaussian": fits a Gaussian (lmfit) and uses the fitted center parameter as the peak.
       - If peak_method == "poly": fits a polynomial (deg=10), then finds the tallest peak via scipy.signal.find_peaks.
       The corresponding fitted curve is optionally overlaid on the measure channel plot.
    3) optionally plots normalized profiles for both channels, aligned by the offset, and overlays:
       - a dashed polynomial or sigmoid fit for the align channel depending on align_method,
       - a dashed Gaussian fit for the measure channel when peak_method == "gaussian"
         (or a dashed polynomial when peak_method == "poly").

    Notes:
    - number_of_channels is forwarded to measure_line_values so that indexing/layout can be handled in vis_tools.
    - channels should be a list of channel names; the returned DataFrame columns will be
      [channels[measure_channel], channels[align_channel]] in that order.
    - Plots are produced as a side-effect (per-ROI profiles and two summary plots: swarm and box).

    Args:
        image_path (list[str]): Paths to the image files.
        roi_path (list[str]): Paths to the corresponding ROI files (.roi or .zip).
        channels (list[str]): Channel display names, e.g., ['DAPI', 'GFP'].
        number_of_channels (int): Total channel count; forwarded to measure_line_values for indexing.
        align_channel (int): 0-based channel index used to compute the alignment offset.
        measure_channel (int): 0-based channel index used to measure the peak position.
        line_width (int, optional): Width (in pixels) of the line profile. Defaults to 5.
        normalize (bool, optional): If True, profiles are min-max normalized for plotting. Defaults to True.
        scaling (float, optional): X-axis scaling factor to convert pixel indices to physical units. Defaults to 0.03525845591290619.
        align (bool, optional): If True, x-axes are shifted by the computed offset for aligned plotting. Defaults to True.
        peak_method (str, optional): "gaussian" (default) or "poly".
        align_method (str, optional): "poly" (default) or "sigmoid" for the half-max offset and overlay.

    Returns:
        pandas.DataFrame: Two-column DataFrame with columns:
            - channels[measure_channel]: peak position of the measure channel relative to the offset (scaled units).
            - channels[align_channel]: zero reference for the align channel (should be ~0 in scaled units).

    Raises:
        ValueError: If ROI files are unsupported or other input validation fails downstream.
    """
    # canvas for per-ROI profile plots
    _, axs = plt.subplots(1, 1, figsize=(10, 5))
    image_peaks = [[], []]  # [measure_offsets, align_offsets]

    for single_image, single_roi in zip(image_path, roi_path):
        roi = read_roi(single_roi)
        image = io.imread(single_image)

        cmap = ListedColormap(['limegreen', 'magenta'])
        color_for = {measure_channel: cmap.colors[0], align_channel: cmap.colors[1]}

        for _, item in roi.items():
            img_slice = item["position"]["slice"]
            src = (item["y1"], item["x1"])
            dst = (item["y2"], item["x2"])

            # 1) Offset from align_channel via chosen half-maximum method
            values_align_channel = measure_line_values(
                image, align_channel, img_slice - 1, src, dst, line_width, number_of_channels
            )
            offset, t_hi, vals_hi = half_max_offset(values_align_channel, method=align_method)

            # 2) Peak from measure_channel using the chosen method
            value_peak_channel = measure_line_values(
                image, measure_channel, img_slice - 1, src, dst, line_width, number_of_channels
            )
            peak_point, gaussian_fit_result = peak_calling(value_peak_channel, method=peak_method)

            # 3) Plot normalized, optionally aligned profiles for both channels
            for channel in (align_channel, measure_channel):
                color = color_for[channel]
                channel_max = []

                if channel == align_channel:
                    # By construction, this evaluates to ~0 in scaled units
                    channel_max.append((t_hi[np.argmax(_safe_minmax(vals_hi) >= 0.5)] - offset) * scaling)

                if channel == measure_channel:
                    channel_max.append(((peak_point if np.isfinite(peak_point) else np.nan) - offset) * scaling)

                # Extract values for plotting this channel
                value_channel = measure_line_values(
                    image, channel, img_slice - 1, src, dst, line_width, number_of_channels
                )
                if normalize:
                    if align:
                        axs.plot(
                            (np.arange(0, len(value_channel)) - offset) * scaling,
                            _safe_minmax(value_channel),
                            color=color,
                        )
                        if channel == align_channel:
                            # Use the same smoothed curve produced by half_max_offset for overlay.
                            # If align_method == "poly": this is the polynomial smooth.
                            # If align_method == "sigmoid": this is the sigmoid fit.
                            linestyle = '--' if align_method == "poly" else ':'
                            axs.plot(
                                (t_hi - offset) * scaling,
                                _safe_minmax(vals_hi),
                                color=color,
                                linestyle=linestyle,
                                alpha=0.9,
                                linewidth=1.5,
                            )

                        if channel == measure_channel and peak_method == "gaussian" and gaussian_fit_result is not None:
                            # Gaussian overlay via lmfit (measure channel)
                            t_plot = np.linspace(0, len(value_channel) - 1, len(value_channel) * 3)
                            gaussian_fit = gaussian_fit_result.eval(x=t_plot)
                            axs.plot((t_plot - offset) * scaling, _safe_minmax(gaussian_fit),
                                     color=color, linestyle='--', alpha=0.9, linewidth=1.5)
                        elif channel == measure_channel and peak_method == "poly":
                            # Polynomial overlay for measure channel for visual consistency
                            poly_meas_plot = np.poly1d(np.polyfit(np.arange(0, len(value_channel)), value_channel, 10))
                            t_plot = np.linspace(0, len(value_channel) - 1, len(value_channel) * 3)
                            vals_plot = poly_meas_plot(t_plot)
                            axs.plot((t_plot - offset) * scaling, _safe_minmax(vals_plot),
                                     color=color, linestyle='--', alpha=0.6, linewidth=1.0)

                        axs.set_xlim(-2, 3.5)
                    else:
                        axs.plot(
                            np.arange(0, len(value_channel)),
                            _safe_minmax(value_channel),
                            color=color,
                        )
                        axs.set_xlim(-2, 3.5)
                else:
                    axs.plot(np.arange(0, len(value_channel)), value_channel, color=color)
                    axs.set_xlim(-2, 3)

                # Collect into image_peaks arrays after plotting
                if channel == measure_channel:
                    image_peaks[0].extend(channel_max)
                if channel == align_channel:
                    image_peaks[1].extend(channel_max)

    # Build DataFrame with stable column order [measure, align]
    df = pd.DataFrame(image_peaks).transpose()
    df.columns = [channels[i] for i in [measure_channel, align_channel]]

    # Summary plots
    fig, axs = plt.subplots(1)
    if align_channel == 0:
        sns.swarmplot(data=df.iloc[:, ::-1])
    else:
        sns.swarmplot(data=df)
    fig, axs = plt.subplots(1)
    if align_channel == 0:
        sns.boxplot(data=df.iloc[:, ::-1])
    else:
        sns.boxplot(data=df)
    return df


def half_max_offset(values_align_channel, method="poly", poly_degree=10, upsample_factor=10):
    """
    Compute alignment offset as the first half-maximum crossing after smoothing.

    Methods:
        - "poly": fit a polynomial (degree=poly_degree), evaluate on a dense grid, take first crossing at 0.5.
        - "sigmoid": fit a sigmoid via lmfit, evaluate on a dense grid, take first crossing at 0.5.

    Args:
        values_align_channel (array-like): Raw profile of the align channel.
        method (str): "poly" (default) or "sigmoid".
        poly_degree (int): Degree of polynomial used for smoothing (poly method only).
        upsample_factor (int): Multiplier for dense sampling.

    Returns:
        tuple: (offset, t_hi, vals_hi)
            - offset (float): x-position (in pixel index units) where the first half-maximum is reached.
            - t_hi (np.ndarray): high-resolution x grid used for evaluation.
            - vals_hi (np.ndarray): smoothed values on t_hi (not normalized).
    """
    y = np.asarray(values_align_channel, dtype=float)
    x = np.arange(0, len(y))
    t_hi = np.linspace(0, len(y) - 1, max(1, upsample_factor) * max(1, len(y)))

    if method == "sigmoid":
        def sigmoid(xx, amplitude, center, sigma, offset0):
            return amplitude / (1 + np.exp(-(xx - center) / sigma)) + offset0

        # Initial guesses
        amp0 = float(np.nanmax(y) - np.nanmin(y)) if np.isfinite(np.nanmax(y) - np.nanmin(y)) else 1.0
        center0 = len(y) / 2.0
        sigma0 = max(1.0, len(y) / 10.0)
        offset0 = float(np.nanmin(y)) if np.isfinite(np.nanmin(y)) else 0.0

        sig_model = Model(sigmoid)
        params = sig_model.make_params(amplitude=amp0, center=center0, sigma=sigma0, offset0=offset0)
        try:
            result = sig_model.fit(y, params, x=x)
            vals_hi = result.eval(x=t_hi)
        except Exception:
            # Fallback to polynomial if sigmoid fit fails
            poly = np.poly1d(np.polyfit(x, y, poly_degree))
            vals_hi = poly(t_hi)
    else:
        poly = np.poly1d(np.polyfit(x, y, poly_degree))
        vals_hi = poly(t_hi)

    vals_norm = _safe_minmax(vals_hi)
    idx = int(np.argmax(vals_norm >= 0.5))  # first index where normalized value exceeds half-max
    offset = t_hi[idx]
    return offset, t_hi, vals_hi


def peak_calling(value_peak_channel, method="gaussian"):
    """
    Estimate the peak location for a 1D profile.

    Methods:
        - "gaussian": lmfit GaussianModel; returns the fitted center parameter as the peak.
        - "poly": degree-10 polynomial + scipy.signal.find_peaks (tallest peak).

    Args:
        value_peak_channel (array-like): Raw profile from the measure channel.
        method (str): "gaussian" (default) or "poly".

    Returns:
        tuple: (peak_point, fit_result)
            - peak_point (float): x-position (pixel index units along the line), np.nan if not found.
            - fit_result: lmfit.ModelResult if method == "gaussian" and fit succeeded; otherwise None.
    """
    y = np.asarray(value_peak_channel, dtype=float)
    x = np.arange(0, len(y))

    if method == "gaussian":
        gauss_model = models.GaussianModel()
        params = gauss_model.guess(y, x=x)
        try:
            result = gauss_model.fit(y, params, x=x)
            mu = float(result.best_values.get("center", np.nan))
            return mu, result
        except Exception:
            return float("nan"), None

    # "poly" fallback
    poly = np.poly1d(np.polyfit(x, y, 10))
    t = np.linspace(0, len(y) - 1, len(y))
    y_sm = poly(t)
    if not np.any(np.isfinite(y_sm)):
        return float("nan"), None
    try:
        peaks, heights = signal.find_peaks(y_sm, height=np.nanmax(y_sm) * 0.6)
        peak_heights = heights.get("peak_heights", [])
        if len(peak_heights) > 0:
            best_idx = int(np.argmax(peak_heights))
            return float(t[peaks[best_idx]]), None
        return float("nan"), None
    except Exception:
        return float("nan"), None


def _safe_minmax(arr):
    arr = np.asarray(arr, dtype=float)
    amin, amax = np.nanmin(arr), np.nanmax(arr)
    denom = (amax - amin)
    if denom == 0 or not np.isfinite(denom):
        return np.zeros_like(arr, dtype=float)
    return (arr - amin) / denom