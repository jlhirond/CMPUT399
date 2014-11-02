CMPUT399
========

Topics in computing science - Visual recognition

--------
jlhirond
--------
Vl_feat must be included within the project folder, it is expected to be in a root folder called VLFEATROOT. Positive and negative images may be added from the files provided by the project description page.

I have modified the data and code from 'practical-category-recognition-2013a\exercise 2' so that it takes in a subset of the images consisting of positive and negative images and outputs images in the same way as the October 27 lab example. For those of you that weren't there, basically the program would more accurately find the positive test images based on the number and variety of training images provided.

With that being said, the results are very wrong because of the histogram data is incorrect for the bear images (recycled from the lab's horse images). I had not figured out what is required for the histograms data yet and spent way too much time trying to get it running. 

Currently, the program uses 100 negative and 50 positive images as test data. It uses all of the bear close ups as positive training data. The code currently requires hardcoded data and would be faster to prototype with if we could get code which automatically creates name and histogram data based on the images present in the 'images' folder. I went with the mentioned lab as a template as it allowed for much modularity and was doing something pretty close to the project requirement. Of course, displaying the images and showing the graph would need to be converted to simply displaying what number of positive images that were found to be above a threshold.