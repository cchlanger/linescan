# linescan
A tool that does a linescan and performs peak detection and alignment based on FIJI ROIs.
## getting started
Open Git :)
## functions
### `plot_line_profiles(image_path, roi_path, number_of_channels, line_width=1)`

This function plots each linescan along with the corresponding image slice it was drawn on.  Individual channels are represented by different colors, and the drawn line is overlaid on the image. The line profile is normalized. Future enhancements may include fitting and peak calling display.  Values are represented in pixels.

**Functionality:**

The `plot_line_profiles` function visualizes line profiles extracted from image data based on specified ROIs.  It's designed to provide a quick and informative way to inspect the intensity variations along lines defined in FIJI, across different channels of an image.

**Parameters:**

* `image_path` (str): The path to the image file.
* `roi_path` (str): The path to the ROI file.
* `number_of_channels` (int): The number of channels in the image.
* `line_width` (int, optional): The width of the line. Defaults to 1.

**Raises:**

* `ValueError`: If the number of channels is not supported (currently only 2 or 3 channels are supported).

## Linescan Analysis Function

### `linescan(image_path, roi_path, channels, number_of_channels, align_channel, measure_channel, line_width=5, normalize=True, scaling=0.03525845591290619, align=True)`

This function performs linescan analysis on images based on provided regions of interest (ROIs). It analyzes line profiles across specified channels, with options for alignment and normalization.  Currently, it supports 2-channel and 3-channel images (3-channel analysis uses a 2-channel implementation by default, analyzing only the `align_channel` and `measure_channel`).

**Functionality:**

The function reads image and ROI data, extracts line profiles for the specified channels, and optionally performs alignment and normalization.  The alignment is based on a designated `align_channel`, shifting profiles to align peaks. Normalization scales the intensity values within each profile. The results, including peak locations, are returned in a Pandas DataFrame.

**Parameters:**

* **`image_path`** (list): A list of paths to the image files (e.g., TIFF, OME-TIFF).
* **`roi_path`** (list): A list of paths to the corresponding ROI files (e.g., `.roi`, `.zip`).  The order of ROI files must match the order of image files.
* **`channels`** (list): A list of channel names corresponding to the image channels (e.g., `['DAPI', 'GFP']`).  This is used for labeling the output DataFrame columns.
* **`number_of_channels`** (int): The number of channels in the images (2 or 3).
* **`align_channel`** (int): The index (0-based) of the channel used for alignment.  Line profiles will be shifted to align based on the peak in this channel.
* **`measure_channel`** (int): The index (0-based) of the channel on which measurements (peak finding) are performed.
* **`line_width`** (int, optional): The width of the line profile in pixels. Defaults to 5.
* **`normalize`** (bool, optional): Whether to normalize the line profiles. Defaults to `True`.
* **`scaling`** (float, optional): Scaling factor for the x-axis (e.g., for converting pixels to physical units). Defaults to 0.03525845591290619.
* **`align`** (bool, optional): Whether to align the line profiles based on the `align_channel`. Defaults to `True`.


**Returns:**

* **`pandas.DataFrame`**: A DataFrame containing the linescan data.  Columns correspond to the channel names provided in `channels`.  The DataFrame includes peak locations for the specified channels, potentially after alignment and normalization.

**Raises:**

* **`ValueError`**: If an unsupported number of channels is provided (i.e., not 2 or 3).

## example notebooks
TODO add notbooks of Claudia and Caelan
## future features
- add pip installer
- use lmfit for interpolation and offer gaussian models
- vis_tools
    - display interpolation
- align_tools
    - remove destinction between 2 and 3 channel in the code
    - catch bad user input
    - for aligned_channel calculate and display a liner interpolation between the closest values instead of the fitted model value (which now is always zero)
