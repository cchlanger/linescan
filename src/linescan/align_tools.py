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


def _fit_gaussian_with_robust_init(y, x=None, n_candidates=3, prominence_frac=0.1):
    """
    Fit a 1D Gaussian with a constant baseline using robust initial guesses.

    Args:
        y (array-like): 1D signal values to fit.
        x (array-like or None, optional): Coordinate vector for y. If None, uses np.arange(n).
        n_candidates (int, optional): Number of candidate peak centers (sorted by prominence) to try for initialization.
        prominence_frac (float, optional): Fraction of the signal dynamic range used to set the peak prominence threshold for candidate centers.

    Returns:
        tuple[float, lmfit.model.ModelResult | None]: A tuple (center, fit_result) where:
            - center is the fitted Gaussian center (in x units), or np.nan on failure.
            - fit_result is the lmfit ModelResult on success, else None.
    """
    # Adds a constant baseline to the model. If your traces ride on a nonzero background, this is often the main reason guess() fails.
    # Uses Savitzky–Golay smoothing plus peak prominence to choose a good initial center; sigma is derived from the measured FWHM; amplitude is set from height×sigma×sqrt(2π).
    # Tries multiple candidate peaks (by prominence) and keeps the best fit, which is much more stable on multi-peak or noisy profiles.
    # Uses a robust loss (“soft_l1”) in the optimizer to blunt the effect of outliers.
    y = np.asarray(y, dtype=float)
    n = len(y)
    if x is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float)

    if n < 3 or not np.any(np.isfinite(y)):
        return float("nan"), None

    # Replace non-finite values by linear interpolation
    mask = np.isfinite(y)
    if not np.all(mask):
        if mask.any():
            y = np.interp(np.arange(n, dtype=float), np.flatnonzero(mask), y[mask])
        else:
            return float("nan"), None

    # Robust baseline and detrend
    b0 = float(np.nanpercentile(y, 10))
    y0 = y - b0

    # Smooth to stabilize peak detection (skip if too short)
    if n >= 7:
        # window ~20% of signal length, odd, capped
        w = max(5, (n // 5) * 2 + 1)
        w = min(w, n - 1 if (n - 1) % 2 == 1 else n - 2)
        if w < 5:
            w = 5 if n >= 5 else (3 if n >= 3 else 1)
        polyorder = min(3, w - 2) if w >= 5 else 2
        try:
            y_s = signal.savgol_filter(y0, window_length=w, polyorder=polyorder, mode="interp")
        except Exception:
            y_s = y0
    else:
        y_s = y0

    # Peak candidates by prominence
    dyn = float(np.nanmax(y_s) - np.nanmin(y_s)) if np.any(np.isfinite(y_s)) else 0.0
    prom = dyn * float(prominence_frac)
    try:
        peaks, props = signal.find_peaks(y_s, prominence=prom if np.isfinite(prom) and prom > 0 else None)
    except Exception:
        peaks, props = np.array([], dtype=int), {"prominences": np.array([])}

    if len(peaks) == 0:
        # Fallback: use global maximum
        peak_idx = int(np.nanargmax(y_s))
        peaks = np.array([peak_idx], dtype=int)
        props = {"prominences": np.array([max(y_s[peak_idx], 0.0)])}

    # Sort candidates by prominence (desc) and keep top-k
    prominences = props.get("prominences", np.zeros(len(peaks)))
    order = np.argsort(prominences)[::-1][: max(1, int(n_candidates))]

    # Estimate width at half-height for candidates
    try:
        widths, _, _, _ = signal.peak_widths(y_s, peaks, rel_height=0.5)
    except Exception:
        widths = np.full(len(peaks), max(3.0, n / 10.0), dtype=float)

    # Build model: Constant baseline + Gaussian
    model = models.ConstantModel(prefix="b_") + models.GaussianModel(prefix="g_")
    best_result = None
    best_score = np.inf

    amp_area_cap = max(1.0, (np.nanmax(y) - np.nanmin(y)) * max(5.0, n / 5.0))

    for idx in order:
        pk = int(peaks[idx])
        height0 = max(float(y_s[pk]), 1e-12)  # height above baseline (non-negative)
        width0 = float(widths[idx]) if np.isfinite(widths[idx]) and widths[idx] > 0 else max(3.0, n / 10.0)
        sigma0 = max(width0 / 2.355, 0.1)     # convert FWHM to sigma, clamp minimum
        center0 = float(x[pk])

        # lmfit GaussianModel uses 'amplitude' as area under the curve
        amp0 = float(height0 * sigma0 * np.sqrt(2.0 * np.pi))
        amp0 = min(max(amp0, 1e-12), amp_area_cap)

        params = model.make_params(
            b_c=b0,
            g_amplitude=amp0,
            g_center=center0,
            g_sigma=sigma0,
        )
        # Reasonable bounds
        params["g_center"].set(min=float(x[0]), max=float(x[-1]))
        params["g_sigma"].set(min=0.1, max=max(2.0, float(x[-1] - x[0])))
        params["g_amplitude"].set(min=0.0, max=amp_area_cap)

        try:
            # Use robust loss to reduce outlier influence
            result = model.fit(
                y,
                params,
                x=x,
                nan_policy="omit",
                method="least_squares",
                fit_kws={"loss": "soft_l1", "f_scale": 0.5},
            )
            score = result.aic if np.isfinite(result.aic) else result.chisqr
            if score < best_score:
                best_score = score
                best_result = result
        except Exception:
            continue

    if best_result is None:
        return float("nan"), None

    center = float(best_result.best_values.get("g_center", np.nan))
    return center, best_result


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
    align_method="sigmoid",   # DEFAULT NOW "sigmoid" (was "poly")
    plot_mode="both",         # "raw", "fit", or "both"
):
    """
    Perform linescan analysis on images using ROI line segments.

    For each ROI line segment, this function:
    1) extracts the line profile from the align_channel and computes an alignment offset as the
       first half-maximum crossing on a smoothed curve:
       - If align_method == "sigmoid": fit a sigmoid (via lmfit), evaluate densely, then find the first 0.5 crossing.
       - If align_method == "poly": fit a degree-10 polynomial, upsample, then find the first 0.5 crossing.
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
        align_method (str, optional): "sigmoid" (default) or "poly" for the offset estimation and align overlay.
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
            # Use the exact half-max used for offset; align channel should be strictly zero
            image_peaks[1].append(0.0)
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

    # Summary plots: measured channel only
    meas_col = channels[measure_channel]
    if meas_col in df.columns:
        fig, ax = plt.subplots(1)
        sns.swarmplot(data=df[[meas_col]])
        fig, ax = plt.subplots(1)
        sns.boxplot(data=df[[meas_col]])
    return df


def _plot_roi_profiles(
    ax,
    align_values,
    measure_values,
    offset,
    scaling,
    normalize=True,
    align=True,
    align_method="sigmoid",   # DEFAULT NOW "sigmoid"
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
    - align_method: "sigmoid" (sigmoid fit) or "poly" (polynomial smooth) via provided t_hi/vals_hi.
    - peak_method: "gaussian" (Gaussian lmfit) or "poly" (polynomial smooth).

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on.
        align_values (array-like): Raw align-channel profile.
        measure_values (array-like): Raw measure-channel profile.
        offset (float): Offset (pixels) for alignment.
        scaling (float): X-axis scaling factor to physical units.
        normalize (bool, optional): If True, min-max normalize y for plotting.
        align (bool, optional): If True, shift x by offset before scaling.
        align_method (str, optional): "sigmoid" or "poly" for align overlay.
        peak_method (str, optional): "gaussian" or "poly" for measure overlay.
        t_hi (np.ndarray or None, optional): High-res x used for align fit overlay (from half_max_offset).
        vals_hi (np.ndarray or None, optional): Smoothed align values for overlay (from half_max_offset).
        gaussian_fit_result (lmfit.model.ModelResult or None, optional): Fit result for Gaussian overlay (measure).
        color_align (str, optional): Color for align channel.
        color_measure (str, optional): Color for measure channel.
        plot_mode (str, optional): "raw", "fit", or "both".

    Returns:
        None
    """
    def _x_axis(n):
        x = np.arange(0, n)
        return ((x - offset) * scaling) if align else x

    def _normalize_with_range(y, lo, hi):
        y = np.asarray(y, dtype=float)
        denom = (hi - lo)
        if denom == 0 or not np.isfinite(denom):
            return np.zeros_like(y, dtype=float)
        return (y - lo) / denom

    # Establish per-channel normalization ranges from the RAW curves
    align_raw = np.asarray(align_values, dtype=float)
    measure_raw = np.asarray(measure_values, dtype=float)
    a_lo = np.nanmin(align_raw) if np.any(np.isfinite(align_raw)) else 0.0
    a_hi = np.nanmax(align_raw) if np.any(np.isfinite(align_raw)) else 1.0
    m_lo = np.nanmin(measure_raw) if np.any(np.isfinite(measure_raw)) else 0.0
    m_hi = np.nanmax(measure_raw) if np.any(np.isfinite(measure_raw)) else 1.0

    def _plot_series(y, color, style='-', use_align_scale=True):
        if normalize:
            yy = _normalize_with_range(y, a_lo, a_hi) if use_align_scale else _normalize_with_range(y, m_lo, m_hi)
        else:
            yy = np.asarray(y, dtype=float)
        ax.plot(_x_axis(len(yy)), yy, color=color, linestyle=style)

    # Raw curves
    if plot_mode in ("raw", "both"):
        _plot_series(align_values, color_align, style='-', use_align_scale=True)
        _plot_series(measure_values, color_measure, style='-', use_align_scale=False)

    # Fit overlays (normalized using the same raw min/max per channel)
    if plot_mode in ("fit", "both"):
        # Align overlay
        if t_hi is not None and vals_hi is not None:
            x_fit = ((t_hi - offset) * scaling) if align else t_hi
            y_fit = _normalize_with_range(vals_hi, a_lo, a_hi) if normalize else vals_hi
            linestyle = '--' if align_method == "poly" else ':'
            ax.plot(x_fit, y_fit, color=color_align, linestyle=linestyle, alpha=0.9, linewidth=1.5)

        # Measure overlay
        if peak_method == "gaussian" and gaussian_fit_result is not None:
            t_plot = np.linspace(0, len(measure_values) - 1, max(3, len(measure_values) * 3))
            gaussian_fit = gaussian_fit_result.eval(x=t_plot)
            x_fit = ((t_plot - offset) * scaling) if align else t_plot
            y_fit = _normalize_with_range(gaussian_fit, m_lo, m_hi) if normalize else gaussian_fit
            ax.plot(x_fit, y_fit, color=color_measure, linestyle='--', alpha=0.9, linewidth=1.5)
        elif peak_method == "poly":
            n = len(measure_values)
            xv = np.arange(0, n, dtype=float)
            deg = min(10, max(1, n - 1))
            try:
                poly_meas = np.poly1d(np.polyfit(xv, measure_values, deg))
                t_plot = np.linspace(0, n - 1, max(3, n * 3))
                vals_plot = poly_meas(t_plot)
            except Exception:
                # Fallback: interpolate the raw values on a dense grid
                t_plot = np.linspace(0, n - 1, max(3, n * 3))
                vals_plot = np.interp(t_plot, xv, np.asarray(measure_values, dtype=float))
            x_fit = ((t_plot - offset) * scaling) if align else t_plot
            y_fit = _normalize_with_range(vals_plot, m_lo, m_hi) if normalize else vals_plot
            ax.plot(x_fit, y_fit, color=color_measure, linestyle='--', alpha=0.6, linewidth=1.0)

    if align:
        ax.set_xlim(-2, 3.5)


def half_max_offset(values_align_channel, method="sigmoid", poly_degree=10, upsample_factor=10):
    """
    Compute alignment offset as the half-maximum crossing.

    For method == "sigmoid", return the fitted center parameter (exact half-max crossing).
    For method == "poly", smooth with a polynomial, then find the first half-max crossing
    using linear interpolation on a dense grid (rising crossings only).

    Args:
        values_align_channel (array-like): Raw profile of the align channel.
        method (str, optional): "sigmoid" (default) or "poly".
        poly_degree (int, optional): Degree of polynomial used for smoothing (poly method only).
        upsample_factor (int, optional): Multiplier for dense sampling for overlays and crossings.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: (offset, t_hi, vals_hi) where:
            - offset is the half-maximum crossing position in pixel index units.
            - t_hi is the dense x grid for overlay plotting.
            - vals_hi are the smoothed values on t_hi for overlay plotting.
    """
    y = np.asarray(values_align_channel, dtype=float)
    x = np.arange(0, len(y))
    n = len(y)
    if n == 0:
        return 0.0, np.array([]), np.array([])

    t_hi = np.linspace(0, n - 1, max(1, upsample_factor) * max(1, n))

    if method == "sigmoid":
        # 4-parameter logistic: offset0 + amplitude / (1 + exp(-(x - center)/sigma))
        def sigmoid(xx, amplitude, center, sigma, offset0):
            return offset0 + amplitude / (1.0 + np.exp(-(xx - center) / sigma))

        # Initial guesses
        y_min, y_max = np.nanmin(y), np.nanmax(y)
        amp0 = float(y_max - y_min) if np.isfinite(y_max - y_min) and (y_max > y_min) else 1.0
        center0 = n / 2.0
        sigma0 = max(1.0, n / 10.0)
        offset0 = float(y_min) if np.isfinite(y_min) else 0.0

        sig_model = Model(sigmoid)
        params = sig_model.make_params(amplitude=amp0, center=center0, sigma=sigma0, offset0=offset0)

        try:
            result = sig_model.fit(y, params, xx=x)  # independent var name matches function
            # Exact half-max crossing for this model is the fitted 'center'
            center = float(result.best_values.get("center", center0))
            vals_hi = result.eval(xx=t_hi)
            offset = center
            return offset, t_hi, vals_hi
        except Exception:
            # Fall back to polynomial smoothing if fitting fails
            pass

    # Polynomial path (either requested or sigmoid fit failed)
    poly_degree = min(poly_degree, max(1, n - 1))  # keep degree < n
    try:
        poly = np.poly1d(np.polyfit(x, y, poly_degree))
        vals_hi = poly(t_hi)
    except Exception:
        # Degenerate fallback: interpolate the raw values on a dense grid
        vals_hi = np.interp(t_hi, x, y)

    # Compute half-level from smoothed curve and find first crossing with interpolation
    a, b = float(np.nanmin(vals_hi)), float(np.nanmax(vals_hi))
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        # give up gracefully
        return float(t_hi[0]), t_hi, vals_hi

    half_level = a + 0.5 * (b - a)

    # Find indices where vals_hi crosses half_level (rising only)
    v = np.asarray(vals_hi, dtype=float)
    above = v >= half_level
    crossings = np.where((~above[:-1]) & (above[1:]))[0]

    if crossings.size == 0:
        # If no rising crossing found, pick the closest point to half-level as a fallback
        idx = int(np.argmin(np.abs(v - half_level)))
        return float(t_hi[idx]), t_hi, vals_hi

    i = crossings[0]
    # Linear interpolation between t_hi[i]..t_hi[i+1]
    y0, y1 = v[i], v[i + 1]
    x0, x1 = t_hi[i], t_hi[i + 1]
    if y1 == y0 or not np.isfinite(y1 - y0):
        return float(x0), t_hi, vals_hi
    frac = (half_level - y0) / (y1 - y0)
    offset = float(x0 + frac * (x1 - x0))
    return offset, t_hi, vals_hi


def peak_calling(value_peak_channel, method="gaussian"):
    """
    Estimate the peak location for a 1D profile.

    Methods:
        - "gaussian": robust init + baseline using lmfit (returns fitted center).
        - "poly": degree-capped polynomial + upsampled peak search (tallest peak).

    Args:
        value_peak_channel (array-like): Raw profile from the measure channel.
        method (str, optional): "gaussian" (default) or "poly".

    Returns:
        tuple[float, lmfit.model.ModelResult | None]: (peak_point, fit_result) where:
            - peak_point is the x-position (pixel index units) of the detected peak or np.nan on failure.
            - fit_result is the lmfit ModelResult when method == "gaussian" and fit succeeded; otherwise None.
    """
    y = np.asarray(value_peak_channel, dtype=float)
    x = np.arange(0, len(y), dtype=float)

    if method == "gaussian":
        mu, result = _fit_gaussian_with_robust_init(y, x)
        return mu, result

    # "poly" path (robust)
    n = len(y)
    if n < 2:
        return float("nan"), None
    deg = min(10, max(1, n - 1))
    try:
        poly = np.poly1d(np.polyfit(x, y, deg))
    except Exception:
        return float("nan"), None

    # Evaluate on a denser grid to refine peak position
    t = np.linspace(0, n - 1, max(3, n * 3))
    y_sm = poly(t)
    if not np.any(np.isfinite(y_sm)):
        return float("nan"), None
    try:
        height_thresh = np.nanmax(y_sm) * 0.6 if np.isfinite(np.nanmax(y_sm)) else None
        peaks, props = signal.find_peaks(y_sm, height=height_thresh)
        if len(peaks) == 0:
            return float("nan"), None
        # Select tallest peak among detected peaks
        peak_heights = props.get("peak_heights", np.array([]))
        best = int(peaks[np.argmax(peak_heights)]) if len(peak_heights) else int(peaks[0])
        return float(t[best]), None
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