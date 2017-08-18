# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[chessboard_before]: ./output_images/chessboard_calibration_image.jpg "Uncalibrated chessboard"
[chessboard_after]: ./output_images/distortion_corrected_chessboard.jpg "Calibrated chessboard"
[road_calibration_before]: ./output_images/road_uncalibrated.jpg "Uncalibrated road"
[road_calibration_after]: ./output_images/road_calibrated.jpg "Calibrated road"
[filter_l]: ./output_images/filter_l.jpg "HSL Lightness filter"
[filter_s]: ./output_images/filter_s.jpg "HSL Saturation filter"
[filter_sobel]: ./output_images/filter_sobel.jpg "Sobel X filter"
[filter_combined]: ./output_images/filter_combined.jpg "Combined filters"
[warped_image]: ./output_images/warped_image.jpg "Warped road image"
[result]: ./output_images/result.jpg "Plotted result image"
[polyfit]: ./output_images/polyfit.jpg "Plotted result image in topdown view"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in the `calibrate_camera` method of project.py. A chessboard calibration image from the `camera_cal/` directory is passed into the method, which is then converted to grayscale. The grayscale image is passed to OpenCV's `cv2.findChessboardCorners` method, along with the given board size information.

Assuming corners are found, the corners are used as image points and an object points matrix is constructed based on what the 2d chessboard for that image size and board size should look like. The object, image points, and input image are passed to `cv2.calibrateCamera`, which then returns calibration information `mtx` and `dist`. These values are used in the pipeline for warping car camera feed images to remove lens warping.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![Chessboard before calibration][chessboard_before]

The shape is warped closer to a flat 2d representation of a chessboard:

![Chessboard before after][chessboard_after]

This is done by the `calibrate_camera` method in `project.py`, which uses OpenCV's `findChessboardCorners` and `calibrateCamera` method to output distortion-correct information. This information is then applied to an image using the `cv2.undistort` method.

Here's the same process applied to a road image, using the calibration data obtained from the above transformation.

![Road before][road_calibration_before]

![Road after][road_calibration_after]


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

My perspective transform is handled in `process_frame`, and uses hard-coded constants along with the input frame size to compose source and destination warp matrices.

The source and destination points with the example inputs are as follows:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 684, 450      | 270, 0        | 
| 1030, 677     | 270, 720      |
| 280, 677      | 490, 720      |
| 598, 450      | 490, 0        |

The image below was used to verify that the perspective transform was working as expected.

![Warped road image][warped_image]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

My project used a combination of the following filters:
* HLS Saturation threshold `20 < S <= 255`
* HLS Lightness threshold `65 < L <= 255`
* Sobel X threshold `2 < sobelx <= 255`
* Cropping out 28% of the image from the left and right sides to reduce noise from other lanes

The filter code can be found in `project.py#thresholded_all`. This function takes an image and applies the abovementioned transforms to generate a binary output where white represents a pixel that passed thresholding and black everything else.

These filters were chosen because saturation was not sufficient for eliminating false lines from barriers or shadows. Lightness thresholding is especially helpful for eliminating dark lines from the accepted pixels.

The results of each filter can be seen here:

Saturation filter

![saturation filter][filter_s]

Lightness filter

![lightness filter][filter_l]

Sobel filter

![sobel filter][filter_sobel]

Combined filter

![combined filter][filter_combined]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane lines were fit using the combined filter image below piped through `find_lane_lines`. This function utilizes the histogram method taught in the module.

Specifically, the function creates a histogram of the bottom half of the filtered binary image and starts searching the left and right halves of the histogram for lane lines. The search is performed using sliding windows that grab non-zero pixels from the binary image starting from the bottom and moving up. The windows are moved left and right whenever a significant number of pixels are found in a given -- in this case, the windows are placed in the center of the found pixel mass. When both lanes lines are detected, np.polyfit is used to fit a second-order polynomial to the pixels found by the sliding windows for the left and right lane lines. If the fit lane lines pass the sanity checks described below, the lane lines will be added to the lane fit queue.

##### Sanity checks

Because lane lines are sometimes improperly detected, a few sanity checks were implemented. If these sanity checks fail, the lane line is rejected for this frame.

* Assert that the lane line X-intercepts are at least 250 pixels apart in the top-down image
* Assert that the lane line X-intercepts are no more than 550 pixels apart in the top-down image

This was found to be effective in rejecting poor fits when the real lane lines were not detected at all.

![Top-down polynomial fit of road lines after filters][polyfit]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature calculation is in `project.py#measure_curvature` and the vehicle position code is in `project.py#calculate_vehicle_position`

The radius of curvature is calculated by taking the polynomial fit for the left and right lane lines and dividing the square of the first derivative by the absolute value of the second derivative, in addition to some constants. Since we want the curvature in world measurements (meters) instead of pixels, we fit a new line against the raw pixels multiplied by the ratio of pixels to meters given in the lecture (derived based on U.S. standards for lane line distances) and use this new polynomial to calculate the radius using the beforementioned formula. The left and right curvature values are then averaged, since the lane lines should ideally be parallel.

The vehicle position is calculated by finding the midpoint between the x-intercepts of the best fit lane line polygons we have and subtracting the midpoint of the image (i.e. half the width) from it. We then use the ratios given in the module to convert pixels to meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is implemented at the end of `project.py#process_frame` by taking the rendered curve polygon image and warping it with the inverse of the camera-to-topdown warp, then overlaying it translucently over the input image.

![result image with lane plotted on road][result]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My project mostly followed the steps outlined in the module. However, I struggled to find parameters that worked in all frames of the project video. The two sections of the video where the car drives over a bridge has a very low contrast between the white lane lines and the road, and therefore required a much lower saturation channel threshold than was acceptable for other frames of the video. But losing the saturation threshold caused too much noise for the cleaner sections of the video, so cropping was utilized on the left and right sides of the top-down transformed image to strip out false positive lane lines such as from barriers on the side of the road or other lane lines.

Because of the low saturation threshold used, lightness and sobel fitlers are utilized to filter out potential noise. Lightness thresholding was especially helpful in rejecting shadows as false lane lines. To reduce jitter, the lane line coefficients were averaged over 6 frames.

However, adding all the above filters created an immense amount of hyperparameters to tune for the project, and it was easy to accidentally overfit the parameters to work on a set of frames but not generally over the entire video. Likewise, it is clear from the challenge video (output found in `challenge_video_output.mp4`) that the pipeline is unable to generalize to other videos even with the same camera.

The project is likely to fail on any video with substantially different lighting, different lane types such as [Botts' dots](https://en.wikipedia.org/wiki/Botts%27_dots), or three-dimensional warps of the road lanes, such as with hills through curves.

To improve this, I considered increasing the average window size but using a function other than mean that would be resilient to single outlier frames. I also considered trying to auto-detect noise levels in a frame and adjust threshold parameters to be more or less aggressive as appropriate. But most of all, I'm curious about using deep learning to detect the lane lines and building a generalized model that would work in almost every condition instead of trying to manually tune parameters based on test videos. This would be similar to the [approach used by Comma.ai's OpenPilot](https://commacoloring.herokuapp.com/).
