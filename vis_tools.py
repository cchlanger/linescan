"""Contains all of the visualization tools for the line_scan application"""

from read_roi import read_roi_zip, read_roi_file
import matplotlib.pyplot as plt
from skimage import io, measure


def min_max_normalization(x):
    return (x - min(x)) / (max(x) - min(x))


def plot_line_profiles(image_path, roi_path, number_of_channels, line_width=1):
    """Plots every every linescan together with the image slice it was drawn on.
    Colors represent the individual channels, the line drawn is diplayed on top of the image.
    The line profiles can be normalized, and the fitting and peak calling can be displayed.
    Values are in pixel unless the user provides the pixel size in microns."""
    if number_of_channels == 2 or number_of_channels == 3:
        # get roi
        if number_of_channels == 2:
            disp_channels = [0, 1]
        if number_of_channels == 3:
            disp_channels = [0, 1, 2]

        if roi_path.endswith(".zip"):
            roi = read_roi_file(roi_path)
        elif roi_path.endswith(".roi"):
            roi = read_roi_file(roi_path)
        else:
            raise ValueError(f"Your ROI file must be a .zip or a .roi!")
        # Counts how many images need to be displayed
        slice_count = 0
        for _, item in roi.items():
            slice_count += 1

        _, axs = plt.subplots(
            max(1, slice_count),
            1 + len(disp_channels),
            figsize=(15, 5 * slice_count),
            squeeze=False,
        )
        image = io.imread(image_path)

        for item_num, item in enumerate(roi.items()):
            _, item = item
            img_slice = item["position"]["slice"]
            src = (item["y1"], item["x1"])
            dst = (item["y2"], item["x2"])
            cmap = plt.get_cmap("tab10")
            # Draw values
            for channel in range(number_of_channels):
                new_color = cmap.colors[channel]
                if number_of_channels == 2:
                    values = measure.profile_line(
                        # slice - 1, because FIJI starts counting at 1
                        image[img_slice - 1, channel, :, :],
                        src,
                        dst,
                        line_width,
                        mode="constant",
                    )
                elif number_of_channels == 3:
                    values = measure.profile_line(
                        image[img_slice - 1, :, :, channel],
                        src,
                        dst,
                        line_width,
                        mode="constant",
                    )
                else:
                    raise ValueError(
                        f"Your channel number: {number_of_channels} is not supported."
                    )
                axs[item_num, 0].plot(
                    range(len(values)), min_max_normalization(values), color=new_color
                )
            # Draw images with lines for the set of channels to display
            for disp_num, disp_channel in enumerate(disp_channels):
                new_color = cmap.colors[disp_num]
                if number_of_channels == 2:
                    axs[item_num, 1 + disp_num].imshow(
                        image[img_slice - 1, disp_channel, :, :], cmap="gray"
                    )
                elif number_of_channels == 3:
                    axs[item_num, 1 + disp_num].imshow(
                        image[img_slice - 1, :, :, disp_channel], cmap="gray"
                    )
                else:
                    raise ValueError(
                        f"Your channel number: {number_of_channels} is not supported."
                    )
                x_values = [item["x1"], item["x2"]]
                y_values = [item["y1"], item["y2"]]
                axs[item_num, 1 + disp_num].plot(x_values, y_values, color=new_color)
    else:
        print(f"{number_of_channels} channels are not supported in this version.")
