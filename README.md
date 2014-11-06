CMPUT399
========

Topics in computing science - Visual recognition

--------
jlhirond
--------
11/1/14
Vl_feat must be included within the project folder, it is expected to be in a root folder called VLFEATROOT. Positive and negative images may be added from the files provided by the project description page.

I have modified the data and code from 'practical-category-recognition-2013a\exercise 2' so that it takes in a subset of the images consisting of positive and negative images and outputs images in the same way as the October 27 lab example. For those of you that weren't there, basically the program would more accurately find the positive test images based on the number and variety of training images provided.

With that being said, the results are very wrong because of the histogram data is incorrect for the bear images (recycled from the lab's horse images). I had not figured out what is required for the histograms data yet and spent way too much time trying to get it running. 

Currently, the program uses 100 negative and 50 positive images as test data. It uses all of the bear close ups as positive training data. The code currently requires hardcoded data and would be faster to prototype with if we could get code which automatically creates name and histogram data based on the images present in the 'images' folder. I went with the mentioned lab as a template as it allowed for much modularity and was doing something pretty close to the project requirement. Of course, displaying the images and showing the graph would need to be converted to simply displaying what number of positive images that were found to be above a threshold.

11/5/14
 As discussed with Abhineet, the histogram data needed for use by this template is not automatically created. Therefore, I had to modify the solution to Assignment 2 to create this histogram data for a test set of bear/bridge images. 

Currently, I use 3 main scripts to do what needs to be done. The main.m script calls constructIMDB2.m (original, I know) 3 times to get 3 sets of data. One is for all images, one is for a set of positive images, and one is for a set of negative images, all of which exist in data/images. Once this is done, compareImages.m (exercise2.m with different hard coded data) may be called to do the object categorization which produces a set of images that it thinks contains the features that we are looking for.

Please note that I have hard coded in constructIMDB2.m to resize the images to 25% of their original size to aid in speed (at the cost of effectiveness). I have also hard coded the scripts for the 125 total images which I have testing with. Of course the runtime will take much longer when we use all images at 100% size (a matter of hours), but will hopefully give us a good showing of bear images.

What needs to be done? We need to make sure the program is doing what it is supposed to be doing, while I get an output of bear images, it could just be lucky. We need to alter the compareImages.m output to tell us how many bear images are found (and maybe the file names) rather than showing images, scores, and a graph. We need to run the program at 100% accuracy once these two things are done to get some results!