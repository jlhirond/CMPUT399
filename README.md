CMPUT399
========

Topics in computing science - Visual recognition

The included files are a blending of the Assignment 2 solution and the object categorization lab. 

main.m creates 2 data sets: 1 for negative images in data\images and one for positive images in data\images. These 2 data sets are required by compareImages.m and need to be created before hand. This data has already been created (15 minute runtime) to save time so that compareImages.m may be optimized.
compareImages.m takes the positive and negative data and uses a training set of images populated by images in data\myImages. As a result, the output is a list of images that are found to match the trained features. Each found image is given a score and listed in the window currently.

A  threshold should be defined for which images to count as bear images and instead of showing the graph and the images, we should instead list how many bear images were found and possibly list the file names of the found images.
