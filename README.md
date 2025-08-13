## Linescan

A tool that performs linescan analysis using FIJI ROIs, including alignment at the half-maximum point and peak detection with flexible fitting options.

## Summary

Linescan is a tool to measure the distance of a point to a surface along user-defined FIJI ROI lines. Lines are assumed to be drawn from outside to inside of the surface. The surface position is aligned using a sigmoid fit (default) on the align channel at the half-maximum reference, and the point position is detected with a Gaussian fit (default) on the measure channel. The reported value is the offset between these two positions in physical units.

## Install
```
pip install git+https://github.com/gerlichlab/linescan.git
```
## Visualization Function

### plot_line_profiles(image_path, roi_path, number_of_channels, line_width=1)

Visualizes each linescan together with the image slice it was drawn on. Individual channels are colored and the ROI line is overlaid on the image. Line profiles are min-max normalized for display. Values are shown in pixels.

Functionality: The function extracts and plots line profiles per ROI for each displayed channel, helping you quickly inspect intensity variations along FIJI-defined lines. Future enhancements may include showing fits and peak calls.

Parameters: image_path (str): Path to the image file. roi_path (str): Path to the ROI file. number_of_channels (int): Number of channels in the image (2 or 3). line_width (int, optional): Line width in pixels. Defaults to 1.

Raises: ValueError: If number_of_channels is not supported (currently only 2 or 3).

## Linescan Analysis Function

### linescan(image_path, roi_path, channels, number_of_channels, align_channel, measure_channel, line_width=5, normalize=True, scaling=0.03525845591290619, align=True, peak_method="gaussian", align_method="sigmoid", plot_mode="both")

Performs linescan analysis on images based on ROIs. For each ROI line segment, it computes an alignment offset from the align_channel at the half-maximum crossing and estimates the peak position in the measure_channel. It can plot raw profiles, fitted overlays, or both, with optional normalization and alignment.

What it does per ROI: 1) Alignment offset (half-maximum): align_method="sigmoid" (default) fits a 4-parameter logistic with lmfit; the fitted center is used as the exact half-max offset. The dense evaluation is used only for overlay plotting. align_method="poly" fits a degree-10 polynomial and finds the first half-max crossing via linear interpolation on a dense grid. The smoothed curve is used for the overlay. 2) Peak detection in the measure channel: peak_method="gaussian" (default) fits a Gaussian with lmfit and uses the fitted center as peak position. The fitted curve is available as an overlay. peak_method="poly" fits a degree-10 polynomial, then finds the tallest peak via scipy.signal.find_peaks. The smoothed curve is available as an overlay. 3) Plotting: plot_mode="raw", "fit", or "both". Y-values can be min-max normalized per channel for visualization. X can be aligned by subtracting the offset and scaled to physical units.

Parameters: image_path (list[str]): Paths to image files. roi_path (list[str]): Paths to corresponding ROI files (.roi or .zip), same order as image_path. channels (list[str]): Channel names, used to label output columns (e.g., ["DAPI", "GFP"]). number_of_channels (int): Total number of channels in the images (2 or 3). align_channel (int): 0-based index of channel used for alignment (half-max offset). measure_channel (int): 0-based index of channel for peak measurement. line_width (int, optional): Line width in pixels. Defaults to 5. normalize (bool, optional): If True, profiles are min-max normalized for plotting. Defaults to True. scaling (float, optional): Factor to scale pixel indices to physical units on the x-axis. Defaults to 0.03525845591290619. align (bool, optional): If True, plots are shifted by the computed offset for alignment. Defaults to True. peak_method (str, optional): "gaussian" or "poly". Defaults to "gaussian". align_method (str, optional): "sigmoid" or "poly". Defaults to "sigmoid". plot_mode (str, optional): "raw", "fit", or "both". Defaults to "both".

Returns: pandas.DataFrame: Two columns in the fixed order [measure, align], where channels[measure_channel] is the peak position of the measure channel relative to the offset (scaled units), and channels[align_channel] is always 0.0 (the align reference), since it is defined relative to the same half-max used as offset.

Notes: Summary plots show only the measured channel (beeswarm and boxplot). The align column is omitted because it is always zero by definition. Fit overlays are normalized using the same min and max as the corresponding raw channel to avoid apparent overshoot.

Raises: ValueError: If ROI format or number_of_channels is unsupported downstream.

## Example notebooks

See repository: https://github.com/gerlichlab/bell_et_al
