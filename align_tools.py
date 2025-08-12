from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from skimage import io
import pandas as pd
import seaborn as sns
import pandas as pd
from pathlib import Path
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
    line_width = 5,
    normalize=True,
    scaling=0.03525845591290619,
    align=True,
):
    """Performs linescan analysis on images based on provided ROIs.

    This function analyzes line profiles across specified channels of images, aligning and normalizing the data if requested.
    It supports both 2-channel and 3-channel images (currently, 3-channel analysis defaults to a 2-channel implementation).

    Args:
        image_path (list): List of paths to the image files.
        roi_path (list): List of paths to the corresponding ROI files.
        channels (list): List of channel names (e.g., ['DAPI', 'GFP']).
        number_of_channels (int): Number of channels in the images (2 or 3).
        align_channel (int): Index of the channel used for alignment (0-based).
        measure_channel (int): Index of the channel to measure (0-based).
        line_width (int, optional): Width of the line profile. Defaults to 5.
        normalize (bool, optional): Whether to normalize the line profiles. Defaults to True.
        scaling (float, optional): Scaling factor for the x-axis. Defaults to 0.03525845591290619.
        align (bool, optional): Whether to align the line profiles. Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame containing the linescan data, with columns corresponding to the channel names.
                            Includes peak locations for the specified channels, potentially after alignment and normalization.

    Raises:
        ValueError: If an unsupported number of channels is provided.
    """
    if number_of_channels == 2:
        result_df = linescan_2c(
            image_path,
            roi_path,
            channels,
            number_of_channels=number_of_channels,
            align_channel=align_channel,
            measure_channel= measure_channel,
            line_width=line_width,
            normalize=normalize,
            scaling=scaling,
            align=align,
        )
        return result_df
    elif number_of_channels == 3:
        # Currently runs the two channel implemention for a pair of measure and align channel
        result_df = linescan_2c(
            image_path,
            roi_path,
            channels,
            number_of_channels=number_of_channels,
            align_channel=align_channel,
            measure_channel=measure_channel,
            line_width=line_width,
            normalize=normalize,
            scaling=scaling,
            align=align,
        )
        return result_df


def linescan_2c(
    image_path,
    roi_path,
    channels,
    number_of_channels,
    align_channel,
    measure_channel,
    line_width,
    normalize,
    scaling,
    align,
):
    # get roi and image

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def find_first_half(b):
        half_lim = max(b) / 2
        for i, y in enumerate(b):
            if y > half_lim:
                break
        return i

    # create plot canvas
    _ , axs = plt.subplots(1, 1, figsize=(10, 5))
    image_peaks = [[], []]
    for single_image, single_roi in zip(image_path, roi_path):
        roi = read_roi(single_roi)
        image = io.imread(single_image)

        # generate a color map iterator
        cmap = ListedColormap(['limegreen', 'magenta'])
        for channel in range(number_of_channels):
            #color = cmap.colors[channel]
            if channel == measure_channel: color = cmap.colors[0]
            if channel == align_channel: color = cmap.colors[1]
            # print(newcolor)
            channel_max = []
            # scaling=1
            for key, item in roi.items():
                img_slice = item["position"]["slice"]
                src = (item["y1"], item["x1"])
                dst = (item["y2"], item["x2"])

                #Calcualte offset for half_maximum
                if channel == measure_channel or channel == align_channel:
                    values_align_channel = measure_line_values(
                        image, align_channel, img_slice - 1, src, dst, line_width, number_of_channels
                    )
                    polinomial = np.poly1d(np.polyfit(np.arange(0, len(values_align_channel)), values_align_channel, 10))
                    number_of_interpolation_points = 10*len(values_align_channel)
                    t = np.linspace(0, max(np.arange(0, len(values_align_channel))), number_of_interpolation_points)
                    values_polinomial = (polinomial(t) - min(polinomial(t))) / (max(polinomial(t)) - min(polinomial(t)))
                    closest = find_first_half(values_polinomial)
                    offset = t[closest]

                #Saves the align_channel - offset = 0
                if channel == align_channel:
                    channel_max.append((t[closest] - offset) * scaling)
                
                #Caluclates the peak in another channel, saves it minus offset
                if channel == measure_channel:
                    value_peak_channel = measure_line_values(
                        image, channel, img_slice -1, src, dst, line_width, number_of_channels
                    )

                    polinomial = np.poly1d(np.polyfit(np.arange(0, len(value_peak_channel)), value_peak_channel, 10))
                    number_of_interpolation_points = len(value_peak_channel)
                    t = np.linspace(
                        0, max(np.arange(0, len(value_peak_channel))), number_of_interpolation_points
                    )
                    peaks, heights = signal.find_peaks(polinomial(t), max(polinomial(t)) * 0.6)
                    heights = heights["peak_heights"].tolist()
                    try:
                        biggest_peak2 = heights.index(max(heights))
                        peak_point = peaks[biggest_peak2]
                    except:
                        peak_point = float("NaN")
                        print(single_roi)
                    channel_max.append((peak_point - offset) * scaling)

                #Plots all the values - normalized and aligned
                if channel == measure_channel or channel == align_channel:
                    value_channel = measure_line_values(
                        image, channel, img_slice - 1, src, dst, line_width, number_of_channels
                    )
                    if normalize == True:
                        if align == True:
                            axs.plot(
                                (np.arange(0, len(value_channel)) - offset) * scaling,
                                (value_channel - min(value_channel)) / (max(value_channel) - min(value_channel)),
                                color=color,
                            )
                            if channel == align_channel:
                                # Create interpolated polynomial for plotting
                                polinomial_plot = np.poly1d(np.polyfit(np.arange(0, len(value_channel)), value_channel, 10))
                                t_plot = np.linspace(0, len(value_channel)-1, len(value_channel)*3)
                                interpolated_values = polinomial_plot(t_plot)
                                interpolated_values_norm = (interpolated_values - min(interpolated_values)) / (max(interpolated_values) - min(interpolated_values))
                                axs.plot((t_plot - offset) * scaling, interpolated_values_norm, color=color, linestyle='--', alpha=0.9, linewidth=1.5)

                                # Define sigmoid function
                                # def sigmoid(x, amplitude, center, sigma, offset):
                                #     return amplitude / (1 + np.exp(-(x - center) / sigma)) + offset


                                # # Fit sigmoid model
                                # x_data = np.arange(0, len(value_channel))
                                # y_data = value_channel

                                # # Create sigmoid model
                                # sigmoid_model = Model(sigmoid)

                                # # Set initial parameters
                                # params = sigmoid_model.make_params(
                                #     amplitude=max(y_data) - min(y_data),
                                #     center=len(y_data) / 2,
                                #     sigma=len(y_data) / 10,
                                #     offset=min(y_data)
                                # )
                                # result = sigmoid_model.fit(y_data, params, x=x_data)
    
                                # # Generate high-resolution x for smooth curve
                                # t_plot = np.linspace(0, len(value_channel)-1, len(value_channel)*3)
                                # sigmoid_fit = result.eval(x=t_plot)
                                # sigmoid_fit_norm = (sigmoid_fit - min(sigmoid_fit)) / (max(sigmoid_fit) - min(sigmoid_fit))
                                # axs.plot((t_plot - offset) * scaling, sigmoid_fit_norm, color=color, linestyle=':', alpha=0.9, linewidth=1.5)

                            if channel == measure_channel:
                                # Fit Gaussian model
                                x_data = np.arange(0, len(value_channel))
                                y_data = value_channel

                                # Create Gaussian model
                                gauss_model = models.GaussianModel()
                                params = gauss_model.guess(y_data, x=x_data)
                                result = gauss_model.fit(y_data, params, x=x_data)
        
                                # Generate high-resolution x for smooth curve
                                t_plot = np.linspace(0, len(value_channel)-1, len(value_channel)*3)
                                gaussian_fit = result.eval(x=t_plot)

                                gaussian_fit_norm = (gaussian_fit - min(gaussian_fit)) / (max(gaussian_fit) - min(gaussian_fit))
                                axs.plot((t_plot - offset) * scaling, gaussian_fit_norm, color=color, linestyle='--', alpha=0.9, linewidth=1.5)


                            axs.set_xlim(-2,3.5)
                            #axs.plot(linewidth=7.0)
                        else:
                            axs.plot(
                                (np.arange(0, len(value_channel))),
                                (value_channel - min(value_channel)) / (max(value_channel) - min(value_channel)),
                                color=color,
                            )
                            axs.set_xlim(-2,3.5)
                          #  axs.plot(linewidth=7.0)
                    else:
                        axs.plot(np.arange(0, len(value_channel)), value_channel, color=color)
                        axs.set_xlim(-2,3)
            if channel == measure_channel: image_peaks[0].extend(channel_max)
            if channel == align_channel: image_peaks[1].extend(channel_max)
            #Shade magenta
            # axs.axvspan(0, max(t_plot) * scaling, facecolor='#DC02D9', alpha=0.2)

    df = pd.DataFrame(image_peaks)
    df = df.transpose()
    channels = [channels[i] for i in [measure_channel,align_channel]]
    df.columns = channels

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
