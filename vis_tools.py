"""Contains all of the visualization tools for the line_scan application"""

import matplotlib.pyplot as plt
from skimage import io, measure
from read_roi import read_roi_file, read_roi_zip


# def plot_image_and_profiles(
#     image_path, roi_path, disp_channels=None, number_of_channels=2
# ):
#     """Plots every z-slice that has at least one linescan.
#     Colors represent the particular line drawn on the image.
#     Channels are displayed one after each other.
#     Channels can be normalized and the fitting and peak calling can be displayed.
#     Further lines can be aligned according to the offset of peak in a praticular channel.
#     """
#     if number_of_channels == 2:
#         plot_image_and_profiles_2c(image_path, roi_path, disp_channels=disp_channels)
#     elif number_of_channels >= 3:
#         plot_image_and_profiles_3c(
#             image_path,
#             roi_path,
#             disp_channels=disp_channels,
#             number_of_channels=number_of_channels,
#         )
#     else:
#         print(
#             f"The channel number {number_of_channels}, is not supported in this version"
#         )


# # TODO: Add 3,4 channel + Error
# # TODO: scaling
# # TODO: label slice in images
# # TODO: allow scaling and
# # TODO: scale along axis of image in microns


# def plot_image_and_profiles_2c(image_path, roi_path, disp_channels=None):
#     """The two channel implementation of plot_image_and_profiles."""
#     # get roi
#     if disp_channels is None:
#         disp_channels = [0, 1]
#     if len(disp_channels) > 3:
#         raise ValueError("You are trying to display more channels then your image has!")
#     number_of_channels = 2
#     if roi_path.find(".zip") == -1:
#         roi = read_roi_file(roi_path)
#     else:
#         roi = read_roi_zip(roi_path)
#     # get slice set
#     slice_set = set()
#     for _, item in roi.items():
#         slice_set.add(item["position"]["slice"])
#     slice_set = sorted(slice_set)

#     _, axs = plt.subplots(
#         len(slice_set),
#         number_of_channels + len(disp_channels),
#         figsize=(15, 5 * len(slice_set)),
#     )
#     image = io.imread(image_path)
#     for slice_num, img_slice in enumerate(slice_set):
#         # Plot linescans of all channels
#         for channel in range(number_of_channels):
#             for _, item in roi.items():
#                 if item["position"]["slice"] == img_slice:
#                     src = (item["y1"], item["x1"])
#                     dst = (item["y2"], item["x2"])
#                     values = measure.profile_line(
#                         image[img_slice - 1, channel, :, :],
#                         src,
#                         dst,
#                         1,
#                         mode="constant",
#                     )
#                     if len(slice_set) > 1:
#                         axs[slice_num, channel].plot(range(len(values)), values)
#                     else:
#                         axs[channel].plot(range(len(values)), values)
#         # Draw images with lines for the set of channels to display
#         for disp_num, disp_channel in enumerate(disp_channels):
#             if len(slice_set) > 1:
#                 axs[slice_num, number_of_channels + disp_num].imshow(
#                     image[img_slice - 1, disp_channel, :, :], cmap="gray"
#                 )
#             else:
#                 axs[number_of_channels + disp_num].imshow(
#                     image[img_slice - 1, disp_channel, :, :], cmap="gray"
#                 )
#             for _, item in roi.items():
#                 if item["position"]["slice"] == img_slice - 1:
#                     x_values = [item["x1"], item["x2"]]
#                     y_values = [item["y1"], item["y2"]]
#                     if len(slice_set) > 1:
#                         axs[slice_num, number_of_channels + disp_num].plot(
#                             x_values, y_values
#                         )
#                     else:
#                         axs[number_of_channels + disp_num].plot(x_values, y_values)


# def plot_image_and_profiles_3c(
#     image_path, roi_path, disp_channels=[0], number_of_channels=3
# ):
#     # get roi
#     roi = read_roi_zip(roi_path)
#     # get slice set
#     slice_set = set()
#     for key, item in roi.items():
#         slice_set.add(item["position"]["slice"])
#     slice_set = sorted(slice_set)
#     # print(number_of_channels)
#     # print(disp_channels)
#     # print(slice_set)
#     fig, axs = plt.subplots(
#         # TODO:Fix this
#         len(slice_set),
#         number_of_channels + len(disp_channels),
#         figsize=(15, 5 * len(slice_set)),
#     )
#     image = io.imread(image_path)
#     for slice_num, img_slice in enumerate(slice_set):
#         # Plot linescans of all channels
#         for channel in range(number_of_channels):
#             for key, item in roi.items():
#                 if item["position"]["slice"] == img_slice:
#                     src = (item["y1"], item["x1"])
#                     dst = (item["y2"], item["x2"])
#                     y = measure.profile_line(
#                         image[img_slice - 1, :, :, channel],
#                         src,
#                         dst,
#                         1,
#                         mode="constant"
#                         # TODO merge with 2c since this is the only difference
#                     )
#                     if len(slice_set) > 1:
#                         axs[slice_num, channel].plot(range(len(y)), y)
#                     else:
#                         axs[channel].plot(range(len(y)), y)
#         # Draw images with lines for the set of channels to display
#         for disp_num, disp_channel in enumerate(disp_channels):
#             if len(slice_set) > 1:
#                 axs[slice_num, number_of_channels + disp_num].imshow(
#                     image[img_slice - 1, :, :, disp_channel], cmap="gray"
#                 )
#             else:
#                 axs[number_of_channels + disp_num].imshow(
#                     image[img_slice - 1, :, :, disp_channel], cmap="gray"
#                 )
#             for key, item in roi.items():
#                 if item["position"]["slice"] == img_slice - 1:
#                     x_values = [item["x1"], item["x2"]]
#                     y_values = [item["y1"], item["y2"]]
#                     if len(slice_set) > 1:
#                         axs[slice_num, number_of_channels + disp_num].plot(
#                             x_values, y_values
#                         )
#                     else:
#                         axs[number_of_channels + disp_num].plot(x_values, y_values)


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
            raise ValueError(
                f"Your ROI file must be a .zip or a .roi!"
            )
        # Counts how many images need to be displayed
        slice_count = 0
        for _, item in roi.items():
            slice_count += 1
        
        _, axs = plt.subplots(
            max(1,slice_count), 1 + len(disp_channels), figsize=(15, 5 * slice_count), squeeze = False
        )
        image = io.imread(image_path)

        for item_num, item in enumerate(roi.items()):
            _, item = item
            img_slice = item["position"]["slice"]
            src = (item["y1"], item["x1"])
            dst = (item["y2"], item["x2"])
            cmap = plt.get_cmap("tab10")
            #Draw values
            for channel in range(number_of_channels):
                new_color = cmap.colors[channel]
                if number_of_channels == 2:
                    values = measure.profile_line(
                        #slice - 1, because FIJI starts counting at 1
                        image[img_slice - 1, channel, :, :], src, dst, line_width, mode="constant"
                    )
                elif number_of_channels == 3:
                    values = measure.profile_line(
                        image[img_slice - 1, :, :, channel], src, dst, line_width, mode="constant"
                    )
                else:
                    raise ValueError(
                        f"Your channel number: {number_of_channels} is not supported."
                    )
                axs[item_num, 0].plot(range(len(values)), min_max_normalization(values), color=new_color)
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
        print(
            f"{number_of_channels} channels are not supported in this version."
        )

    
