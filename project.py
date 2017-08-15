import cv2
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

# Calibrate camera

def calibrate_camera(chess_img, board_size=(8,6)):
    grayscale = grayscale_img(chess_img)
    return_val, corners = cv2.findChessboardCorners(grayscale, board_size, None)

    assert return_val != 0, "Could not find corners."

    image_points = corners
    object_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    object_points[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1, 2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([object_points], [image_points], grayscale.shape[::-1], None, None)

    return mtx, dist

def grayscale_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def undistort_img(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

img = cv2.imread("camera_cal/calibration2.jpg")
#cv2.imshow('img', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

mtx, dist = calibrate_camera(img)

#cv2.imshow('image', undistort_img(img, mtx, dist))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Input frame
# Warp frame into top-down
# Find lane lines (moving window method)
# Draw on image
# Wrap back into normal view

ROAD_TRANSFORM_SRC = np.float32([
        [684, 450],
        [1030, 677],
        [280, 677],
        [598, 450]])

previous_fit = None
previous_frame = None
iterations = 0

def process_frame(frame):
    global previous_fit, previous_frame, iterations
    iterations += 1

    img_size = (frame.shape[1], frame.shape[0])
    ROAD_TRANSFORM_DEST = np.float32([
        [img_size[0] - 490, 0],
        [img_size[0] - 490, img_size[1]],
        [490, img_size[1]],
        [490, 0]])
    warp_matrix = cv2.getPerspectiveTransform(ROAD_TRANSFORM_SRC, ROAD_TRANSFORM_DEST)
    warped_image = cv2.warpPerspective(frame, warp_matrix, img_size, flags=cv2.INTER_LINEAR)

    thresholded = thresholded_saturation(warped_image, s_threshold=(80,255))
    thresholded = topdown_crop(thresholded)

    if iterations > 1:
        # Smear.
        tmp = previous_frame
        previous_frame = thresholded
        thresholded = cv2.addWeighted(thresholded, 1, tmp, 0.5, 0)
    else:
        previous_frame = thresholded

    """
    cv2.imshow('img', thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    #window_centroids = find_window_centroids(thresholded, window_width, window_height, margin)

    #output = draw_centroids(thresholded, window_width, window_height, window_centroids)
    if previous_fit:
        left_fit, right_fit, leftx, rightx, lefty, righty = find_lane_lines_incremental(thresholded, previous_fit)
        print("Left fit", left_fit[0], left_fit[1], left_fit[2])

        current_fit = (left_fit, right_fit)
        if fit_change_acceptable(current_fit, previous_fit):
            previous_fit = current_fit
        else:
            print("Fit NOT acceptable: ", current_fit, previous_fit)
            left_fit, right_fit, output, leftx, rightx, lefty, righty = find_lane_lines(thresholded)

            current_fit = (left_fit, right_fit)
            previous_fit = current_fit
    else:
        left_fit, right_fit, output, leftx, rightx, lefty, righty = find_lane_lines(thresholded)
        previous_fit = (left_fit, right_fit)
        current_fit = (left_fit, right_fit)

    #measure_curvature(left_fit, right_fit, leftx, rightx, lefty, righty, thresholded)

    color_warp = draw_lines(current_fit[0], current_fit[1], thresholded)

    # Unwarp
    warp_matrix = cv2.getPerspectiveTransform(ROAD_TRANSFORM_DEST, ROAD_TRANSFORM_SRC)
    color_unwarp = cv2.warpPerspective(color_warp, warp_matrix, img_size, flags=cv2.INTER_LINEAR)

    result = cv2.addWeighted(frame, 1, color_unwarp, 0.3, 0)
    """
    cv2.imshow('img', thresholded)
    cv2.imshow('img2', color_warp)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    """
    cv2.imshow('img', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    return result

def draw_lines(left_fit, right_fit, img):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    return color_warp

def measure_curvature(left_fit, right_fit, leftx, rightx, lefty, righty, img):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print(left_curverad, right_curverad)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    print(len(ploty), len(leftx))
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')

def thresholded_saturation(img, s_threshold=(0, 255), l_threshold=(50, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    saturation = hls[:,:,2]
    lightness = hls[:,:,1]

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel_dir = np.arctan2(abs_sobely, abs_sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sobel_threshold = (10, 255)
    sobel_dir_threshold = (0, 3 * 3.14 / 2) #3.14 / 6, 11 * 3.14 / 6)

    binary_output = np.zeros_like(saturation)
    binary_output[
            (saturation > s_threshold[0]) & (saturation <= s_threshold[1]) &
            (lightness > l_threshold[0]) & (lightness <= l_threshold[1]) &
            (sobel_dir > sobel_dir_threshold[0]) & (sobel_dir <= sobel_dir_threshold[1])] = 255

    return binary_output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids

window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching
img = mpimg.imread("test_images/straight_lines1.jpg")
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
saturation = hls[:,:,2]
window_centroids = find_window_centroids(saturation, window_width, window_height, margin)

def find_lane_lines_incremental(img, previous_fit):
    left_fit = previous_fit[0]
    right_fit = previous_fit[1]
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each

    if len(leftx) < 1 or len(lefty) < 1 or len(rightx) < 1 or len(righty) < 1:
        print("No lanes found")
        return (0,0,0), (0,0,0), leftx, rightx, lefty, righty

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fit, right_fit, leftx, rightx, lefty, righty

def find_lane_lines(img):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    print(righty)
    print(rightx)
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    out_img = np.dstack((img, img, img))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    return left_fit, right_fit, result, leftx, rightx, lefty, righty

def pct_change(val1, val2):
    diff = abs(val1 - val2)

    if diff == 0: return 0
    return (diff / val2) * 100.0

def fit_change_acceptable(current_fit, previous_fit):
    for i in range(len(current_fit)):
        for j in range(len(current_fit[i])):
            if pct_change(current_fit[i][j], previous_fit[i][j]) > 400:
                return False
    return True

def topdown_crop(img):
    output = np.copy(img)
    
    for i in range(output.shape[0]):
        for j in range(0, 400):
            output[i][j] = 0
        for j in range(output.shape[1] - 400, output.shape[1]):
            output[i][j] = 0
    return output

"""
print(output)
plt.imshow(output)
plt.title('window fitting results')
plt.show()
cv2.imshow('img', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

white_output = 'output.mp4'
clip1 = VideoFileClip("project_video.mp4") #.subclip(26, 32)
white_clip = clip1.fl_image(process_frame)
white_clip.write_videofile(white_output, audio=False)

