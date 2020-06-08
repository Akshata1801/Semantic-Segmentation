# Semantic-Segmentation

This repository contains scripts and model for performing semantic segmentation on CityScapes dataset using pytorch. 
The models folder contain all the models which were used for doing semantic segmentation which are mainly Segnet, Combination of Segnet and U-net that is Segnet with skip Connections wherein Upsampling is performed on downsampled images and then coupled with feature map of corresponding encoder layer. The third model also uses skip connection but the Upsampling is performed using Convulation.
The output of the three models are shown in output folder

The output using Basic Segnet is
<p float="center">
  <img src="../master/output/img1.png" width="300" />
</p>

The output using combination of Segnet and U-net is
<p float="center">
  <img src="../master/output/out_skip_usegnet2.JPG" width="300" />
</p>

The output using Segnet with skip connection and Upsampling using convolution is
<p float="center">
  <img src="../master/output/out_skip_conv_2.JPG" width="300" />
</p>

