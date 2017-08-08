# Behavioral Cloning 

Jonathan Warner, with sources from Udacity CarND nanodegree.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./examples/model.png "Model Visualization"
[tensorboard]: ./examples/tensorboard.png "Tensorboard"
[track1]: ./examples/track_1.jpg "Track 1 driving example"
[track2]: ./examples/track2.jpg "Track 2 driving example"
[track1_leftcenterright]: ./examples/track1_leftcenterright.jpg "Track 1 multiple camera example"
[track1_gif]: ./output/track1.gif "Track 1 driving gif"
[track2_gif]: ./output/track2.gif "Track 2 driving gif"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* WRITEUP.md summarizing the results

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the [NVIDIA end to end deep learning architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

Dropout was added to reduce overfitting the model.

The data is normalized using a lambda layer, and the hood section of the car (bottom 25 pixels) and some tree line (top 70 pixels) are removed to reduce noise that is irrelevant to the driving model. In addition, RELUs are used to introduce non-linearity into the model, as is true with NVIDIA's architecture.

#### 2. Attempts to reduce overfitting in the model

In addition to the dropout layer mentioned above, multiple training datasets were used to train the model, including one where the car was turned around and looped the track backwards.

#### 3. Model parameter tuning

The model used an adam optimizer with default learning rate params from Keras.

#### 4. Appropriate training data

In addition to the sample Udacity data, the training data I created includes:

* 3x Driving track 1 normally
* 2x Driving track 2 normally
* 3x Recovering from off-the-road situations in track 1
* Driving track 1 backwards

Training data was augmented by using left, center, and right images, and flipping each one and the corresponding steering angle. For example, here is the same scene, but shown using left, right, and center images:

![Multiple camera angles in use][track1_leftcenterright]

To adjust the steering angle, a static factor of +-0.2 was utilized, as given in the lecture videos.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow nvidia's proven example for driving a car with E2E deep learning using camera input. This seemed like a appropriate fit because the task at hand is almost identical, just with a simulator instead of real life.

With just the track 1 training data, the model drove decently on track 1, but with some extreme jitter in the steering and an increasing validation error at each epoch. Adding dropout helped combat this.

When testing on track 2, the data from track 1 was not general enough to sufficiently drive the track at all. After training with a single loop of track 2 at 5 epochs, the car drove almost the entire track 2 successfully, but consistently hit a wall on one of the turns. I reloaded the model and trained a single epoch on additional loop of track 2 and the model was then able to sufficiently drive on both track 1 and track 2.

#### 2. Final Model Architecture

![Keras model architecture based on NVIDIA's drive architecture][model]

The details of the model can further be explored in Tensorboard using the included data in the logs directory:

![Screenshot of Tensorboard][tensorboard]

#### 3. Training Process

Including all training data sets, I ended up with 28807 samples.

In the model, this data is preprocessed by normalizing it by dividing each entity by 255 (total color space) and subtracting 0.5. In addition, each image was cropped to remove the top and bottom sections to reduce noise and data size.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

The model was set to train for 10 epochs, but I enabled checkpoint saving in Keras and found that after the first three epochs, the validation error started to climb, which implies overfitting. The `save_best_only` option was enabled to ensure only models with a lower validation error than the previously saved one are retained. I consistently found 3 to be the sweet spot for epochs in my pipeline.

The model results can be seen in the following GIFs, or as videos in the output directory.

![Track 1 animation][track1_gif]
![Track 2 animation][track2_gif]

Track 2 was a particularly interesting challenge because sections of the road are visible in the distance through some of the curves, and the car enters shadows occasionally which sigificantly changes the color of the road. However, with just a single loop of the track, the model was able to generalize enough to complete almost an entire perfect loop.

![Track 2 driving into a shadow][track2]
