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
):
    """
    Perform linescan analysis on images using ROI line segments.

    For each ROI line segment, this function:
    1) extracts the line profile from the align_channel, fits a polynomial (deg=10) and
       finds the first half-maximum crossing on the smoothed curve to define an offset;
    2) extracts the line profile from the measure_channel, fits a polynomial (deg=10),
       finds the most prominent peak, and reports the peak position relative to the offset;
    3) optionally plots normalized profiles for both channels, aligned by the offset, and
       overlays a dashed polynomial fit for the align channel and a dashed Gaussian fit
       for the measure channel (via lmfit).

    Notes:
    - number_of_channels is not used for branching here; it is forwarded to measure_line_values
      so that indexing/layout can be handled centrally in vis_tools.
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

    Returns:
        pandas.DataFrame: Two-column DataFrame with columns:
            - channels[measure_channel]: peak position of the measure channel relative to the offset (scaled units).
            - channels[align_channel]: zero reference for the align channel (should be ~0 in scaled units).

    Raises:
        ValueError: If ROI files are unsupported or other input validation fails downstream.
    """
    def find_first_half(b):
        half_lim = max(b) / 2
        for i, y in enumerate(b):
            if y > half_lim:
                return i
        return int(np.argmax(b))

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

            # Offset from align_channel via first half-maximum on a degree-10 polynomial
            values_align_channel = measure_line_values(
                image, align_channel, img_slice - 1, src, dst, line_width, number_of_channels
            )
            poly_align = np.poly1d(np.polyfit(np.arange(0, len(values_align_channel)), values_align_channel, 10))
            t_hi = np.linspace(0, len(values_align_channel) - 1, 10 * len(values_align_channel))
            vals_hi = poly_align(t_hi)

            # normalize safely to locate half-maximum
            denom = (np.max(vals_hi) - np.min(vals_hi))
            if denom == 0 or not np.isfinite(denom):
                vals_norm = np.zeros_like(vals_hi)
            else:
                vals_norm = (vals_hi - np.min(vals_hi)) / denom
            closest = find_first_half(vals_norm)
            offset = t_hi[closest]

            # Process only align_channel and measure_channel
            for channel in (align_channel, measure_channel):
                color = color_for[channel]
                channel_max = []

                if channel == align_channel:
                    # By construction, this evaluates to ~0 in scaled units
                    channel_max.append((t_hi[closest] - offset) * scaling)

                if channel == measure_channel:
                    value_peak_channel = measure_line_values(
                        image, channel, img_slice - 1, src, dst, line_width, number_of_channels
                    )
                    poly_meas = np.poly1d(np.polyfit(np.arange(0, len(value_peak_channel)), value_peak_channel, 10))
                    t_meas = np.linspace(0, len(value_peak_channel) - 1, len(value_peak_channel))
                    y_meas = poly_meas(t_meas)

                    peaks, heights = signal.find_peaks(y_meas, height=np.max(y_meas) * 0.6)
                    peak_heights = heights.get("peak_heights", [])
                    if len(peak_heights) > 0:
                        best_idx = int(np.argmax(peak_heights))
                        peak_point = t_meas[peaks[best_idx]]
                    else:
                        peak_point = float("NaN")
                        print(single_roi)
                    channel_max.append((peak_point - offset) * scaling)

                # Plot normalized, optionally aligned profiles
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
                            # dashed polynomial overlay (align channel)
                            poly_plot = np.poly1d(np.polyfit(np.arange(0, len(value_channel)), value_channel, 10))
                            t_plot = np.linspace(0, len(value_channel) - 1, len(value_channel) * 3)
                            vals_plot = poly_plot(t_plot)
                            axs.plot((t_plot - offset) * scaling, _safe_minmax(vals_plot),
                                     color=color, linestyle='--', alpha=0.9, linewidth=1.5)

                            # ------------------------------
                            # Optional sigmoid fitting block
                            # ------------------------------
                            # def sigmoid(x, amplitude, center, sigma, offset0):
                            #     return amplitude / (1 + np.exp(-(x - center) / sigma)) + offset0
                            #
                            # x_data = np.arange(0, len(value_channel))
                            # y_data = value_channel
                            #
                            # sigmoid_model = Model(sigmoid)
                            # params = sigmoid_model.make_params(
                            #     amplitude=max(y_data) - min(y_data),
                            #     center=len(y_data) / 2,
                            #     sigma=max(1.0, len(y_data) / 10),
                            #     offset0=min(y_data),
                            # )
                            # result = sigmoid_model.fit(y_data, params, x=x_data)
                            #
                            # t_sig = np.linspace(0, len(value_channel) - 1, len(value_channel) * 3)
                            # sigmoid_fit = result.eval(x=t_sig)
                            # axs.plot((t_sig - offset) * scaling, _safe_minmax(sigmoid_fit),
                            #          color=color, linestyle=':', alpha=0.9, linewidth=1.5)
                            # ------------------------------

                        if channel == measure_channel:
                            # Gaussian overlay via lmfit (measure channel)
                            x_data = np.arange(0, len(value_channel))
                            y_data = value_channel
                            gauss_model = models.GaussianModel()
                            params = gauss_model.guess(y_data, x=x_data)
                            result = gauss_model.fit(y_data, params, x=x_data)
                            t_plot = np.linspace(0, len(value_channel) - 1, len(value_channel) * 3)
                            gaussian_fit = result.eval(x=t_plot)
                            axs.plot((t_plot - offset) * scaling, _safe_minmax(gaussian_fit),
                                     color=color, linestyle='--', alpha=0.9, linewidth=1.5)
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


def _safe_minmax(arr):
    arr = np.asarray(arr, dtype=float)
    amin, amax = np.min(arr), np.max(arr)
    denom = (amax - amin)
    if denom == 0 or not np.isfinite(denom):
        return np.zeros_like(arr)
    return (arr - amin) / denom