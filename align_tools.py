from scipy import signal
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy import ndimage as ndi
import skimage
from skimage import io
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import chirp, find_peaks, peak_widths
import pandas as pd
from pathlib import Path
import scipy.stats
from read_roi import read_roi_zip, read_roi_file
from .vis_tools import measure_line_values, read_roi

def linescan(
    image_path,
    roi_path,
    channels,
    number_of_channels,
    align_channel,
    measure_channel,
    normalize=True,
    scaling=0.03525845591290619,
    align=True,
):
    if number_of_channels == 2:
        result_df = linescan_2c(
            image_path,
            roi_path,
            channels,
            # This is hacked so bad
            align_channel=align_channel,
            normalize=normalize,
            scaling=scaling,
            align=align,
        )
        # print(result_df)
        return result_df
    elif number_of_channels == 3:
        result_df = linescan_3c(
            image_path,
            roi_path,
            channels,
            align_channel=align_channel,
            measure_channel=measure_channel,
            normalize=normalize,
            scaling=scaling,
            align=align,
        )
        # print(result_df)
        return result_df


def linescan_3c(
    image_path,
    roi_path,
    channels,
    align_channel,
    measure_channel,
    scaling,
    normalize=True,
    align=True,
):
    return linescan_2c(
        image_path,
        roi_path,
        channels,
        align_channel,
        normalize=True,
        scaling=scaling,
        align=True,
    )


def linescan_2c(
    image_path,
    roi_path,
    channels,
    align_channel,
    scaling,
    normalize=True,
    align=True,
):
    # get roi and image
    line_width = 5
    number_of_channels = 2
    scaling = 0.03525845591290619

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
        # print(roi)
        image = io.imread(single_image)
        # print(image.shape)
        # plt.imshow(image[10,0,:,:],cmap="gray")

        # generate a color map iterator
        cmap = plt.get_cmap("tab10")
        colors = iter(cmap.colors)

        for channel in range(number_of_channels):
            newcolor = next(colors)
            # print(newcolor)
            channel_max = []
            # scaling=1
            for key, item in roi.items():
                img_slice = item["position"]["slice"]
                src = (item["y1"], item["x1"])
                dst = (item["y2"], item["x2"])
                values_align_channel = measure_line_values(
                    image, align_channel, img_slice -1, src, dst, 5, number_of_channels
                )

                polinomial = np.poly1d(
                    np.polyfit(np.arange(0, len(values_align_channel)), values_align_channel, 10)
                )
                max_number = 50
                t = np.linspace(
                    0, max(np.arange(0, len(values_align_channel))), max_number
                )

                # get highest peak
                peaks, heights = signal.find_peaks(polinomial(t), max(values_align_channel) * 0.6)
                heights = heights["peak_heights"].tolist()
                # biggest_peak = heights.index(max(heights))
                values_align_channel = values_align_channel.tolist()
                # offset = (peaks[biggest_peak]/max_number)*max(np.arange(0,len(y_align))*pixelsize)
                # Hack:
                biggest_peak = values_align_channel.index(max(values_align_channel))

                max_number = len(values_align_channel)

                ##HACK
                # slice - 1, because FIJI starts counting at 1
                values_dna_channel = measure_line_values(
                    image, align_channel, img_slice - 1, src, dst, 5, number_of_channels
                )
                polinomial = np.poly1d(np.polyfit(np.arange(0, len(values_dna_channel)), values_dna_channel, 10))
                max_number = 10000
                # max_number=len(y_align)*1000
                t = np.linspace(0, max(np.arange(0, len(values_dna_channel))), max_number)
                # print(len(t))
                values_polinomial = (polinomial(t) - min(polinomial(t))) / (max(polinomial(t)) - min(polinomial(t)))
                ##print(yy)

                # offset=biggest_peak
                # print("a")
                pt = values_polinomial.tolist()
                # closest = pt.index(find_nearest(yy,max(yy)/2))
                closest = find_first_half(values_polinomial)
                offset = t[closest]
                # def func1(u):
                #    return ((p(u)-min(p(u)))/(max(p(u))-min(p(u))))-(1/2)
                if channel == align_channel:
                    channel_max.append((t[closest] - offset) * scaling)
                    # plt.plot(t[closest]-offset, 0.5, marker='o', markersize=3, color="red")

                if channel != align_channel:
                    y3 = measure_line_values(
                    image, channel, img_slice -1, src, dst, 10, number_of_channels
                )

                    polinomial = np.poly1d(np.polyfit(np.arange(0, len(y3)), y3, 10))
                    max_number = len(y3)
                    t = np.linspace(
                        0, max(np.arange(0, len(y3))), max_number
                    )
                    # axs.plot((t-offset),(y3-min(y3))/(max(y3)-min(y3)),color = "red")
                    # get highest peak
                    peaks, heights = signal.find_peaks(polinomial(t), max(polinomial(t)) * 0.6)
                    # print(peaks)
                    # print(heights)

                    heights = heights["peak_heights"].tolist()
                    try:
                        biggest_peak2 = heights.index(max(heights))
                        peak_point = peaks[biggest_peak2]
                    except:
                        peak_point = float("NaN")
                        print(single_roi)
                    # peak_point = peaks[biggest_peak2]
                    channel_max.append((peak_point - offset) * scaling)
                    # print(peak_point)
                    # plt.plot(peak_point-offset, 1, marker='o', markersize=3, color="red")

                ##end_HACK

                # offset = biggest_peak
                # measure:
                y = skimage.measure.profile_line(
                    image[img_slice - 1, channel, :, :], src, dst, 10, mode="constant"
                )
                if normalize == True:
                    if align == True:
                        axs.plot(
                            (np.arange(0, len(y)) - offset) * scaling,
                            (y - min(y)) / (max(y) - min(y)),
                            color=newcolor,
                        )
                        # plt.hlines(0.5, -2, 2)
                    else:
                        axs.plot(
                            (np.arange(0, len(y))),
                            (y - min(y)) / (max(y) - min(y)),
                            color=newcolor,
                        )
                else:
                    axs.plot(np.arange(0, len(y)), y, color=newcolor)
            image_peaks[channel].extend(channel_max)

    #  print(image_peaks)
    df = pd.DataFrame(image_peaks)
    df = df.transpose()
    df.columns = channels
    # print(df)
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
