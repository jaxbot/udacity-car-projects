# Finding Lane Lines on the Road 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteRight.jpg "Solid white right"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline started with the following series of transformations on the input image frame:

1. Convert to grayscale
2. Run canny to find edges
3. Guassian blur canny edges
4. Crop out our region of interest -- a polygon in the center of the frame forming a trapezoid where the lane lines are expected

The Hough Lines algorithm is then used to retrieve a list of lines from our cropped, transformed image.

These lines are passed into a new method, `average_line`, which also accepts a horizontal cut off to segment left and right lanes. Lines within the specified horizontal segment have their slopes calculated per `m = (y2 - y1) / (x2 - x1)`. This is used to derive a y-intersect, `b`, and extrapolate two x-values for our line. These x-values are added to upper and lower arrays, and averaged together to give us an average line from the bottom to top of the image. This line is averaged with the past three frames to provide a smoothing effect like seen in the example video, and to prevent a single confusing frame from derivating the line significantly.

`draw_lines()` was simplified to draw a 1D array of lines, and is finally called to render two straight lines from the left and right lanes on our image that are derived from extrapolating the hough lines in `road_lines`.

![Pipeline run on solidWhiteRight][image1]

### 2. Identify potential shortcomings with your current pipeline


One shortcoming is that the system relies on contrast between the road lines in the image and the background road. Without a strong contrast, the canny output will not show peaks where the lines are, and the pipeline would fail to annotate them properly. This could be triggered in low light conditions, with faded road lines, or with shadows in the frame.

Another shortcoming is that this pipeline does not support curved lines, which limits the ability to annotate lines in turns at great distances.

This pipeline also does not account for double striped lines (no-passing zones in the USA) or double-dashed lines (reversible lanes).

The pipeline also does not account for vehicles closely in front of the camera blocking road lines.


### 3. Suggest possible improvements to your pipeline

One possible improvement is to discard lines with siginificant derivation from previous frames, to prevent one improperly annotated frame from skewing the next few frames' lane line annotations.

Another improvement would be to look at specific color channels in the image and look for variances there, instead of grayscaling the image and relying on decent contrast between road lines and the road background. This would help resolve issues with low lighting conditions and shadows.
