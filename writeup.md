**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[cnn-classify]: ./output_images/cnn-classify.jpg
[hog-classify]: ./output_images/hog-classify.jpg
[hog-heatmap]: ./output_images/hog-heatmap.jpg
[optical-flow]: ./output_images/optical-flow.jpg
[optical-flow-single]: ./output_images/optical-flow-vehicle.jpg
[orlando-optical-flow-glitch]: ./output_images/orlando-optical-flow-glitch.gif
[pipeline-results-1]: ./output_images/pipeline-results-1.jpg
[pipeline-results-2]: ./output_images/pipeline-results-2.jpg
[pipeline-results-3]: ./output_images/pipeline-results-3.jpg
[vehicle-hog]: ./output_images/hog-vehicles.png
[non-vehicle-hog]: ./output_images/hog-non-vehicles.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
## Writeup / README

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

## Histogram of Oriented Gradients (HOG)

1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The training images are processed in `model.py`, which trains a linear SVM as shown in the lesson. This SVM is then pickled for consumption later in the vehicle detection pipeline.

The `vehicle` and `non-vehicle` training images are loaded and passed into the processing pipeline, which extracts their histograms of oriented gradients (HoG) using code based on the code given in the lesson and Scipy's built-in `sklearn.features.hog` function.

** Vehicle HOG **

![Vehicle HOG][vehicle-hog]

** Non-vehicle HOG **

![Non-vehicle HOG][non-vehicle-hog]

2. Explain how you settled on your final choice of HOG parameters.

My overall goal was to keep the HOG classifier as fast as possible, so I only used the actual gradients without adding color bins. The final acceptance or rejection of a region is determined by the Inception CNN, so adding additional filters was not necessary and would have added more room for error and performance regression.

For the HOG parameters themselves, I picked 11 orientations in the YUV colorspace, using all channels. I found this effective enough for find cars in the frame after experimenting with different parameters in the module's experiment lab.

3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The HOG classifier is trained in `model.py`, which runs independently of the rest of the project and pickles a Linear SVM from the Scipy package.

The classifier is trained by extracting HOG features from all training images using SciPy's builtin `hog` method. These images are then given binomial labels, 0 for non-vehicles and 1 for vehicles. A Linear SVM is used and fits the training data and labels with a 80%/20% split between training and testing data.

## Sliding Window Search

1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My project pipeline used two sliding window searches. The first sliding window search is implemented in `detection.py` and mostly follows the example from the lesson module. I found 1.5x to be a reasonable scale to run a sliding window search on, with images downscaled to 64x64 since the HoG classifier is trained on 64x64 examples. I found that running additional scales did not add any benefit, as the heatmap will return a large enough section for the second sliding window classifier. The windows were overlapped by 32px, which I found to give reasonable results with a speed tradeoff. The windows returned from the first sliding window classifier in the pipeline can be seen below:

![Classifying HoG windows][hog-classify]

For the second part of the pipeline, I took the results of the heatmap and passed it through a retrained Inception v3 CNN to find sections of the heatmap that contain vehicles. The heatmap section is scaled to make the minimal edge 299px, as the input size of the retrained Inception model is 299x299. If the width of the resized window section is greater than 299, as it tends to be when two vehicles are side by side, a second sliding window is used and run against the CNN classifier on the raw pixel data. This second sliding window moves by 64px to the end of the image and adds any boxes that do not overlap, but have a classification probability of greater than 0.8 for vehicles. The results of the second window filter can be seen below:

![Classifying CNN][cnn-classify]

The Inception v3 model is trained using TensorFlow's [built-in Inception retraining script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py) and is given the same training images as the HOG SVM pipeline, but upscaled to the model's required 299x299 input size.

2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![White car classified and tracked by pipeline][pipeline-results-1]

![White and black cars nearby but classified individually][pipeline-results-2]

![Two cars far apart and classified separately][pipeline-results-3]

Most of the performance optimization of this pipeline involved implementing the additional Inception classifier to weed out false positives, as well as runtime optimization by using optical flow to track the vehicles after classification such that the CNN does not need to be run for every frame.

### Optical flow

In my pipeline, once a vehicle is found in a bounding box, optical flow points of interest are found using `cv2.goodFeaturesToTrack`. This method gives us points that we can calculate optical flow using `cv2.calcOpticalFlowPyrLK` that will give us the delta in position of a single point between two frames. The average motion (discarding obvious outliers) is used to pin a bounding box to where we expect the vehicle to remain. We then verify that the vehicle is still within those bounds every frame by running our CNN against the bounding box. If the vehicle is not found within its bounding box after 5 frames, the entire vehicle finding pipeline is run on the frame to try and detect where the lost vehicle has gone. In addition, the pipeline is run every frame that no vehicles are found, or every 15 frames otherwise to find any potential new vehicles.

I found optical flow to reduce jitter, but with some caveats that are noted below in the discussion.

Optical flow tracking points can be seen in the bounding box below. Long tails such as the line on the road are discarded.

![Optical flow on a single vehicle][optical-flow-single]

---

# Video Implementation

1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_images/project-output.mp4)

Here's another attempt on my own dashcam video around my university, which it does not perform as well on (for reasons described in discussion): [UCF output](./output_images/ucf-output.mp4)

2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The first step in my pipeline was to use a heatmap on the bounding boxes returned from the sliding window HoG classifier, which combines any adjacent bounding boxes. This code exists in `detection.py#find_new_vehicles`.

The bounding boxes resulting from the heatmaps are then passed into a retrained Inception v3 CNN classifier. I found this to be a very effective way of eliminating false positives, as well as a reasonable way to continue to track vehicles once they were initially found.

Bounding boxes are only added to the list of tracked vehicles if they do not collide with an existing vehicle bounding box.

---

### Discussion

1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

My approach was hybrid in that I used both the HoG SVM described in the lesson and a CNN classifier for finding vehicles. Running the CNN classifier on every single window would have been too expensive, but I found the HoG classifier to be too prone to falsing and found the CNN that operated on raw pixel values to be easier to work with and had much better accuracy. Since the CNN classifier, which is backed by [Inception](https://arxiv.org/abs/1602.07261), takes about 70ms on my machine to process a single bounding box, it would be far too expensive to scan every possible bounding box on every frame. To mitigate this, I used Optical Flow to track the likely position of the bounding box at the next frame, which is discussed in more detail above.

I've found that my pipeline fails on video taken from my car dashcam, which has a severe fisheye for capturing a wide angle of the road. I believe this is interfering with classifying based on HoG, as vehicles in the edge of the frame are marked as negatives. My pipeline also fails to track optical flow properly when the motion of the camera is in flux, or when the optical flow points chosen include points both parallel and perpendicular to the vehicle's motion. This can be seen in the last seconds of my UCF test video:

![Optical flow bug seen at the end of my UCF, Orlando driving clip, where the bounding box sticks erratically][orlando-optical-flow-glitch]

In addition, this pipeline is likely to fail on night driving videos where the images of cars would be limited to taillights and headlights.

The biggest issue I faced in my implementation was dealing with false positives from the HoG classifier, while still trying to prevent missed classifications. This is why an additional classifier was added for the bounding boxes returned by the HoG heatmap.

To make this pipeline more robust in the future, I want to replace the HoG sliding window scan with an R-CNN like [YOLO](https://arxiv.org/abs/1506.02640) or [Single-Shot-Detection](https://arxiv.org/abs/1512.02325). From demonstrations other students showed, I believe this approach would eliminate the redudancy I currently have implemented by using both an HoG classifier and a raw pixel data classifier, while still achieving the desired performance. I've also seen demonstrations where this approach ran at nearly 30fps, which would bring it close to real time and could be further optimized using optical flow to track vehicles between scanning frames.
