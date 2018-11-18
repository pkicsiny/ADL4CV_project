# ADL4CV_project
Prediction of rainy areas from satellite images with deep learning algorithms.
This site was built using [GitHub Pages](https://pages.github.com/).
Weather data source: [GitHub Pages](ftp://ftp-cdc.dwd.de/pub/CDC/grids_germany/hourly/radolan/historical/asc/)
This is the open data server of the DWD (Deutscher Wetterdienst). Hourly is the highest time resolution available. The files contain rain measurements in 0.1mm in every hour (at XX:50). The measurements cover a 900km x 900km area over Germany and some adjacent areas. Resolution is 1km * 1km. The data is stored in .asc files that can be extracted with the np.loadtxt() numpy method in python. Thus you will get a 900 * 900 large 2D array with each element representing a 1km * 1km grid cell containing the rain height as an integer.
