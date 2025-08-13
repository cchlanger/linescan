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
    number_of_channels,
    align_channel,
    measure_channel,
    line_width=5,
    normalize=True,
    scaling=0.03525845591290619,
    align=True,
    peak_method="gaussian",   # "gaussian" (default) or "poly"
    align_method="poly",      # "poly" (default) or "sigmoid"
    plot_mode="both",         # "raw", "fit", or "both"
):
    """
    Perform linescan analysis on images using ROI line segments.

    For each ROI line segment, this function:
    1) extracts the line profile from the align_channel and computes an alignment offset as the
       first half-maximum crossing on a smoothed curve:
       - If align_method == "poly": fit a degree-10 polynomial, upsample, then find the first 0.5 crossing.
       - If align_method == "sigmoid": fit a sigmoid (via lmfit), evaluate densely, then find the first 0.5 crossing.
       The same smoothed curve is available for plotting as the align overlay.
    2) extracts the line profile from the measure_channel and estimates the peak position:
       - If peak_method == "gaussian": fits a Gaussian (lmfit) and uses the fitted center parameter as the peak.
       - If peak_method == "poly": fits a polynomial (deg=10), then finds the tallest peak via scipy.signal.find_peaks.
       The corresponding fitted curve is available for plotting as the measure overlay.
    3) optionally plots raw data, fitted curves, or both (plot_mode), normalized and optionally aligned by the offset.

    Args:
        image_path (list[str]): Paths to the image files.
        roi_path (list[str]): Paths to the corresponding ROI files (.roi or .zip).
        channels (list[str]): Channel display names, e.g., ['DAPI', 'GFP'].
        number_of_channels (int): Total channel count; forwarded to measure_line_values for indexing.
        align_channel (int): 0-based channel index used to compute the alignment offset.
        measure_channel (int): 0-based channel index used to measure the peak position.
        line_width (int, optional): Width (in pixels) of the line profile.
        normalize (bool, optional): If True, profiles are min-max normalized for plotting.
        scaling (float, optional): X-axis scaling factor to convert pixel indices to physical units.
        align (bool, optional): If True, x-axes are shifted by the computed offset for aligned plotting.
        peak_method (str, optional): "gaussian" (default) or "poly" for the peak estimation and measure overlay.
        align_method (str, optional): "poly" (default) or "sigmoid" for the offset estimation and align overlay.
        plot_mode (str, optional): "raw", "fit", or "both" to control what is drawn.

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

            # Align values and offset
            align_values = measure_line_values(
                image, align_channel, img_slice - 1, src, dst, line_width, number_of_channels
            )
            offset, t_hi, vals_hi = half_max_offset(align_values, method=align_method)

            # Measure values and peak
            measure_values = measure_line_values(
                image, measure_channel, img_slice - 1, src, dst, line_width, number_of_channels
            )
            peak_point, gaussian_fit_result = peak_calling(measure_values, method=peak_method)

            # Collect peak metrics (scaled, relative to offset)
            image_peaks[1].append((t_hi[np.argmax(_safe_minmax(vals_hi) >= 0.5)] - offset) * scaling)  # align ~0
            image_peaks[0].append(((peak_point if np.isfinite(peak_point) else np.nan) - offset) * scaling)

            # Plot this ROI (raw, fit, or both)
            _plot_roi_profiles(
                ax=axs,
                align_values=align_values,
                measure_values=measure_values,
                offset=offset,
                scaling=scaling,
                normalize=normalize,
                align=align,
                align_method=align_method,
                peak_method=peak_method,
                t_hi=t_hi,
                vals_hi=vals_hi,
                gaussian_fit_result=gaussian_fit_result,
                color_align=color_for[align_channel],
                color_measure=color_for[measure_channel],
                plot_mode=plot_mode,
            )

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


def _plot_roi_profiles(
    ax,
    align_values,
    measure_values,
    offset,
    scaling,
    normalize=True,
    align=True,
    align_method="poly",
    peak_method="gaussian",
    t_hi=None,
    vals_hi=None,
    gaussian_fit_result=None,
    color_align="magenta",
    color_measure="limegreen",
    plot_mode="both",  # "raw", "fit", "both"
):
    """
    Plot a single ROI's align and measure profiles on the given axis.

    Raw curves use the input values (optionally normalized and aligned). Fit overlays use:
    - align_method: "poly" (polynomial smooth) or "sigmoid" (sigmoid fit) via provided t_hi/vals_hi.
    - peak_method: "gaussian" (Gaussian lmfit) or "poly" (polynomial smooth).

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on.
        align_values (array-like): Raw align-channel profile.
        measure_values (array-like): Raw measure-channel profile.
        offset (float): Offset (pixels) for alignment.
        scaling (float): X scaling factor to physical units.
        normalize (bool): Min-max normalize the y-values for plotting.
        align (bool): Shift x by offset if True.
        align_method (str): "poly" or "sigmoid" for align overlay.
        peak_method (str): "gaussian" or "poly" for measure overlay.
        t_hi (np.ndarray|None): High-res x used for align fit overlay (from half_max_offset).
        vals_hi (np.ndarray|None): Smoothed align values for overlay (from half_max_offset).
        gaussian_fit_result (lmfit.ModelResult|None): Fit result for Gaussian overlay (measure).
        color_align (str): Color for align channel.
        color_measure (str): Color for measure channel.
        plot_mode (str): "raw", "fit", or "both".

    Returns:
        None
    """
    def _x_axis(n):
        x = np.arange(0, n)
        return ((x - offset) * scaling) if align else x

    def _plot_series(y, color, style='-'):
        yy = _safe_minmax(y) if normalize else np.asarray(y, dtype=float)
        ax.plot(_x_axis(len(yy)), yy, color=color, linestyle=style)

    # Raw curves
    if plot_mode in ("raw", "both"):
        _plot_series(align_values, color_align, style='-')
        _plot_series(measure_values, color_measure, style='-')

    # Fit overlays
    if plot_mode in ("fit", "both"):
        # Align overlay
        if t_hi is not None and vals_hi is not None:
            x_fit = ((t_hi - offset) * scaling) if align else t_hi
            y_fit = _safe_minmax(vals_hi) if normalize else vals_hi
            linestyle = '--' if align_method == "poly" else ':'
            ax.plot(x_fit, y_fit, color=color_align, linestyle=linestyle, alpha=0.9, linewidth=1.5)

        # Measure overlay
        if peak_method == "gaussian" and gaussian_fit_result is not None:
            t_plot = np.linspace(0, len(measure_values) - 1, max(3, len(measure_values) * 3))
            gaussian_fit = gaussian_fit_result.eval(x=t_plot)
            x_fit = ((t_plot - offset) * scaling) if align else t_plot
            y_fit = _safe_minmax(gaussian_fit) if normalize else gaussian_fit
            ax.plot(x_fit, y_fit, color=color_measure, linestyle='--', alpha=0.9, linewidth=1.5)
        elif peak_method == "poly":
            poly_meas = np.poly1d(np.polyfit(np.arange(0, len(measure_values)), measure_values, 10))
            t_plot = np.linspace(0, len(measure_values) - 1, max(3, len(measure_values) * 3))
            vals_plot = poly_meas(t_plot)
            x_fit = ((t_plot - offset) * scaling) if align else t_plot
            y_fit = _safe_minmax(vals_plot) if normalize else vals_plot
            ax.plot(x_fit, y_fit, color=color_measure, linestyle='--', alpha=0.6, linewidth=1.0)

    if align:
        ax.set_xlim(-2, 3.5)


def half_max_offset(values_align_channel, method="poly", poly_degree=10, upsample_factor=10):
    """
    Compute alignment offset as the first half-maximum crossing after smoothing.

    Methods:
        - "poly": fit a polynomial (degree=poly_degree), evaluate on a dense grid, take first crossing at 0.5.
        - "sigmoid": fit a sigmoid via lmfit, evaluate on a dense grid, take first crossing at 0.5.

    Args:
        values_align_channel (array-like): Raw profile of the align channel.
        method (str, optional): "poly" (default) or "sigmoid".
        poly_degree (int, optional): Degree of polynomial used for smoothing (poly method only).
        upsample_factor (int, optional): Multiplier for dense sampling.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: (offset, t_hi, vals_hi)
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
            poly = np.poly1d(np.polyfit(x, y, poly_degree))
            vals_hi = poly(t_hi)
    else:
        poly = np.poly1d(np.polyfit(x, y, poly_degree))
        vals_hi = poly(t_hi)

    vals_norm = _safe_minmax(vals_hi)
    idx = int(np.argmax(vals_norm >= 0.5))
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
        method (str, optional): "gaussian" (default) or "poly".

    Returns:
        tuple[float, lmfit.model.ModelResult|None]:
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
    """
    Normalize an array to [0, 1] using min-max scaling, robust to non-finite values.

    Args:
        arr (array-like): Input array.

    Returns:
        np.ndarray: Min-max normalized array with non-finite handling.
    """
    arr = np.asarray(arr, dtype=float)
    amin, amax = np.nanmin(arr), np.nanmax(arr)
    denom = (amax - amin)
    if denom == 0 or not np.isfinite(denom):
        return np.zeros_like(arr, dtype=float)
    return (arr - amin) / denom