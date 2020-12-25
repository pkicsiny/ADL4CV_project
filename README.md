# ADL4CV_project
This is a project in the subject "Advanced Deep Learning for Computer Vision".

The goal is the prediction of rainy areas from radar images with deep generative networks.

Weather data source: [Link](https://opendata.dwd.de/weather/radar/composit/rx/)
<br>(Note: this server is continuously updated and contains the files for measurements of the past 48 hours. Older data is deleted.
<br>See details at: [Link](https://www.dwd.de/DE/leistungen/opendata/hilfe.html?nn=16102&lsbId=625220) (first file, bottom of page 4))

The files contain rain measurements converted to unitless pixel intensities (0-255). To get back the values in mm/h, use the following formulas: [1](https://www.dwd.de/DE/leistungen/radolan/radolan_info/radolan_radvor_op_komposit_format_pdf.pdf?__blob=publicationFile&v=11) (page 10), [2](https://web.archive.org/web/20160113151652/http://www.desktopdoppler.com/help/nws-nexrad.htm#rainfall%20rates).

The radar maps are recorded with 5 minutes frequency. The maps are uniformly masked and cover a 900km x 900km area over Germany and some adjacent areas. Spatial resolution is 1km * 1km per pixel. The data is stored in binary files and once downloaded, it can be loaded with the src.get_data() method. We use the radar maps to randomly crop 64 X 64 pixel frame sequences and validate them to prevent having masked parts, empty and overly saturated frames. This dataset is then used as input for the networks.

The rough idea of our approach is depicted below.

<p align="center">
  <img src=plots/idea.png>
</p>
  
The generator is a U-net and it gets a sequence of consecutive frames and outputs the next frame at timestamp t+1. The first (spatial) discriminator gets the whole sequence as input with the prediction or ground truth appended to it. The second (temporal) discriminator uses optical flow frames precalculated with the Lukas-Kanade method:

<p align="center">
  <img src=plots/optical_flow_1.png>
</p>

to warp the last frame of the input into a "physically correct" representation of the next frame by then applying the second order discretized advection (warp) equation:

<p align="center">
  <img src=plots/advection_2.png>
</p>

This is then appended to the prediction/ground truth and fed into the discriminator as input. The two discriminators only differ in their inputs but have identical architecture.

The images below show some example predictions from the test set by iteratively using two frames as input to predict the following frame. Predictions with the models developed by us, the baseline model and the ground truth sequence are shown.

<p align="center">
  <img src=plots/comparison_1.png width="700">
</p>

<p align="center">
  <img src=plots/comparison_2.png width="700">
</p>
