# Traffic Sign Recognition 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image_augment]: ./examples/augment_example.jpg "Augmented image"
[gts-1]: ./signs-from-web/german_sign_1.jpg "GTS1"
[gts-2]: ./signs-from-web/german_sign_2.jpg "GTS2"
[gts-4]: ./signs-from-web/german_sign_4.jpg "GTS4"
[gts-7]: ./signs-from-web/german_sign_7.jpg "GTS7"
[gts-8]: ./signs-from-web/german_sign_8.jpg "GTS8"
[exploration]: ./examples/explore.png "Dataset exploration"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jaxbot/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Pandas was utilized to generate the following analysis, found in the ipynb file:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set that demonstrates through histograms that the distribution is consistent between test, validation, and training datasets.

![distribution of training, validation, and test classes][exploration]

The most common signs in the dataset are:

```
#1 most common: Speed limit (50km/h) w/ 2010
#2 most common: Speed limit (30km/h) w/ 1980
#3 most common: Yield w/ 1920
#4 most common: Priority road w/ 1890
#5 most common: Keep right w/ 1860
```

### Design and Test a Model Architecture

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the images as required using X - mean(X) / range(X). I opted to stick to full color images in my pipeline.

To augment the dataset, I first tried increasing the contrast of the images, but quickly realized this would not truly augment my dataset as the contrast would be normalized anyway to match the original normalized image. Instead, I decided to augment the dataset by zooming in every image by 1.25 and 1.5 times and cropping back to 32x32. The goal here is to reduce the pipeline's dependency on the size of a sign in any given training image.

Here is an example of an original image and an augmented image, along with their normalized forms:

![An original image, an augmented image at 1.5x scale, and the normalized versions of each.][image_augment]

The augmented data set contains all of the original data set, plus 1.25x and 1.5x scales for each image. All images are normalized in both sets.

2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description                       | 
|:-----------------------:|:---------------------------------------------:| 
| Input                   | 32x32x3 RGB image                             | 
| Convolution 5x5     	  | 1x1 stride, same padding, outputs 28x28x12    | 
| RELU                    |                                               |
| Max pooling	      	  | 2x2 stride, VALID, outputs 14x14x12           |
| Convolution 5x5         | 1x1 stride, same padding, outputs 10x10x16    |
| RELU                    |                                               |
| Max Pooling             | 2x2 stride, same padding, outputs 5x5x16      |
| Fully connected flatten | outputs 400                                   |
| Dropout                 |                                               |
| Fully connected matmul  | outputs 120                                   |
| RELU                    |                                               |
| Fully connected matmul  | outputs 84                                    |
| Fully connected matmul  | outputs 43 (# of classes)                     |


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 128 with 100 epochs and a learning rate of 0.001. The LeNet architecture described above was optimized using an Adam Optimizer to reduce the cost defined by the mean softmax cross entropy between the given and predicted labels.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.941
* validation set accuracy of 0.947
* test set accuracy of 0.938

My approach was to start with the LeNet architecture described in the handwriting lab, as I figured this would be a reasonable start for an image classifier. Both the handwriting problem and the traffic signs problem take fixed-sized images in for training and testing, so I figured adapting the given example would be a reasonable approach. I adjusted the number of channels to support 3-channel RGB images instead of greyscale.

I spent some time iterating on the model and augmenting the dataset, but from the get-go the vanilla LeNet architecture achieved over 90% accuracy on the validation set, so I felt confident that this approach would work.

In the future I would consider AlexNet, as I've done explorations outside this class with transfer learning on that architecture and it seemed to work well for general purpose image classification.

### Test a Model on New Images

1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![no entry sign][gts-1] ![bumpy road sign][gts-2] ![no passing sign][gts-4] 
![keep right sign][gts-7] ![keep right sign][gts-8]

The last two images may pose interesting challenges for this classifier as one is a cartoon stock image of the sign and differs a bit from the format of the training and validation data, and the other is distorted slightly and is off center, unlike the data set.

The other images are fairly similar to the dataset, aside from differences in crop centering.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction  			| 
|:---------------------:|:---------------------------------------------:| 
| No entry     		| No entry  					| 
| Bumpy road            | Bumpy road					|
| No passing            | No passing     				|
| Keep right     	| Keep right					|
| Keep right     	| Keep right					|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.938.


Visualizations can be seen in the jupyter notebook.
####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

As seen in the jupyter notebook, the softmax probabilities are ~1.0 for the selected guess and ~0 for the subsequent options. I am unsure why this is the case and what the correlation is to the hyper parameters.

```
Image 1: No entry=1.0
Image 1: No passing for vehicles over 3.5 metric tons=1.1441229347730972e-25
Image 1: Traffic signals=1.7535462692330398e-33
Image 1: Speed limit (20km/h)=0.0
Image 1: Speed limit (30km/h)=0.0
Image 2: Bumpy road=1.0
Image 2: Speed limit (20km/h)=0.0
Image 2: Speed limit (30km/h)=0.0
Image 2: Speed limit (50km/h)=0.0
Image 2: Speed limit (60km/h)=0.0
Image 3: No passing=1.0
Image 3: No passing for vehicles over 3.5 metric tons=1.6797080764775575e-29
Image 3: Speed limit (20km/h)=0.0
Image 3: Speed limit (30km/h)=0.0
Image 3: Speed limit (50km/h)=0.0
Image 4: Keep right=1.0
Image 4: Speed limit (20km/h)=0.0
Image 4: Speed limit (30km/h)=0.0
Image 4: Speed limit (50km/h)=0.0
Image 4: Speed limit (60km/h)=0.0
Image 5: Keep right=1.0
Image 5: Speed limit (20km/h)=0.0
Image 5: Speed limit (30km/h)=0.0
Image 5: Speed limit (50km/h)=0.0
Image 5: Speed limit (60km/h)=0.0
```

A barchart illustrating this can be seen in the jupyter notebook.
