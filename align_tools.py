from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import pandas as pd
import seaborn as sns
import pandas as pd
from pathlib import Path
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
        cmap = plt.get_cmap("tab10")
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
                        else:
                            axs.plot(
                                (np.arange(0, len(value_channel))),
                                (value_channel - min(value_channel)) / (max(value_channel) - min(value_channel)),
                                color=color,
                            )
                    else:
                        axs.plot(np.arange(0, len(value_channel)), value_channel, color=color)
            if channel == measure_channel: image_peaks[0].extend(channel_max)
            if channel == align_channel: image_peaks[1].extend(channel_max)

    df = pd.DataFrame(image_peaks)
    df = df.transpose()
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
