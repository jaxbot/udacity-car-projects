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

CAMERA_MTX, CAMERA_DIST = calibrate_camera(img)

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
    global LEFT_LANE_LINE, RIGHT_LANE_LINE, CAMERA_MTX, CAMERA_DIST

    img_size = (frame.shape[1], frame.shape[0])

    frame = undistort_img(frame, CAMERA_MTX, CAMERA_DIST)

    ROAD_TRANSFORM_DEST = np.float32([
        [img_size[0] - 490, 0],
        [img_size[0] - 490, img_size[1]],
        [490, img_size[1]],
        [490, 0]])
    warp_matrix = cv2.getPerspectiveTransform(ROAD_TRANSFORM_SRC, ROAD_TRANSFORM_DEST)
    warped_image = cv2.warpPerspective(frame, warp_matrix, img_size, flags=cv2.INTER_LINEAR)

    thresholded = thresholded_all(warped_image, s_threshold=(15,255))
    thresholded = topdown_crop(thresholded)

    """
    cv2.imshow('img', thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    if len(LEFT_LANE_LINE.fits) > 0 and False:
        LEFT_LANE_LINE, RIGHT_LANE_LINE = find_lane_lines_incremental(thresholded, LEFT_LANE_LINE, RIGHT_LANE_LINE)
    else:
        LEFT_LANE_LINE, RIGHT_LANE_LINE = find_lane_lines(thresholded, LEFT_LANE_LINE, RIGHT_LANE_LINE)

    #measure_curvature(left_fit, right_fit, leftx, rightx, lefty, righty, thresholded)

    average_lines([LEFT_LANE_LINE, RIGHT_LANE_LINE])

    color_warp = draw_lines(LEFT_LANE_LINE, RIGHT_LANE_LINE, thresholded)

    # Unwarp
    warp_matrix = cv2.getPerspectiveTransform(ROAD_TRANSFORM_DEST, ROAD_TRANSFORM_SRC)
    color_unwarp = cv2.warpPerspective(color_warp, warp_matrix, img_size, flags=cv2.INTER_LINEAR)

    result = cv2.addWeighted(frame, 1, color_unwarp, 0.3, 0)
    cv2.imshow('img', thresholded)
    cv2.imshow('img_warped', warped_image)
    cv2.imshow('img_l', thresholded_lightness(warped_image))
    cv2.imshow('img_sobel', thresholded_sobel(warped_image))
    cv2.imshow('img_s', thresholded_saturation(warped_image))
    cv2.imshow('img2', color_warp)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    cv2.imshow('img', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    return result

def average_lines(lines):
    for line in lines:
        best_fit = (0, 0, 0)
        for fit in line.fits:
            best_fit = (best_fit[0] + fit[0], best_fit[1] + fit[1], best_fit[2] + fit[2])
        best_fit = (best_fit[0] / len(line.fits), best_fit[1] / len(line.fits), best_fit[2] / len(line.fits))
        line.best_fit = best_fit
        print("Len", len(line.fits), line.best_fit)

        if len(line.fits) > 4 and line.detected:
            line.fits.pop(0)

def draw_lines(left_lane, right_lane, img):
    left_fit = left_lane.best_fit
    right_fit = right_lane.best_fit

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

def thresholded_all(img, s_threshold=(20, 255), l_threshold=(50, 255)):
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

    sobel_threshold = (2, 255)
    sobel_dir_threshold = (0, 3 * 3.14 / 2) #3.14 / 6, 11 * 3.14 / 6)

    binary_output = np.zeros_like(saturation)
    binary_output[
            (saturation > s_threshold[0]) & (saturation <= s_threshold[1]) &
            (lightness > l_threshold[0]) & (lightness <= l_threshold[1]) &
            (scaled_sobel > sobel_threshold[0]) & (scaled_sobel <= sobel_threshold[1])] = 255

    return binary_output

def thresholded_saturation(img, s_threshold=(15, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    saturation = hls[:,:,2]

    binary_output = np.zeros_like(saturation)
    binary_output[
            (saturation > s_threshold[0]) & (saturation <= s_threshold[1])] = 255

    return binary_output

def thresholded_sobel(img, sobel_threshold = (4, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel_dir = np.arctan2(abs_sobely, abs_sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sobel_dir_threshold = (0, 3 * 3.14 / 2) #3.14 / 6, 11 * 3.14 / 6)

    binary_output = np.zeros_like(gray)
    binary_output[(scaled_sobel > sobel_threshold[0]) & (scaled_sobel <= sobel_threshold[1])] = 255

    return binary_output

def thresholded_lightness(img, l_threshold = (50, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lightness = hls[:,:,1]

    binary_output = np.zeros_like(saturation)
    binary_output[
            (lightness > l_threshold[0]) & (lightness <= l_threshold[1])] = 255

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

def find_lane_lines_incremental(img, left_lane_line, right_lane_line):
    left_fit = left_lane_line.best_fit
    right_fit = right_lane_line.best_fit
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
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
        left_lane_line.detected = False
        right_lane_line.detected = False
        return left_lane_line, right_lane_line

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_lane_line.detected = True
    right_lane_line.detected = True

    #left_lane_line.diffs = left_lane_line.current_fit - left_fit
    left_lane_line.current_fit = left_fit
    left_lane_line.fits.append(left_fit)
    left_lane_line.allx = leftx
    left_lane_line.ally = lefty
    #right_lane_line.diffs = right_lane_line.current_fit - right_fit
    right_lane_line.current_fit = right_fit
    right_lane_line.fits.append(right_fit)
    right_lane_line.allx = rightx
    right_lane_line.ally = righty

    if left_lane_line.best_fit is None:
        left_lane_line.best_fit = left_fit
    if right_lane_line.best_fit is None:
        right_lane_line.best_fit = right_fit

    return left_lane_line, right_lane_line

def find_lane_lines(img, left_lane_line, right_lane_line):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)

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
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
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

    if len(leftx) < 1 or len(lefty) < 1 or len(rightx) < 1 or len(righty) < 1:
        left_lane_line.detected = False
        right_lane_line.detected = False

        print("Could not detect lane!")
    else:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        """
        print(left_fit[1])
        print(right_fit[1])

        print(float(left_fit[1]) * float(right_fit[1]))

        if float(left_fit[1]) * float(right_fit[1]) > 0:
            print("Discarding!")
            left_lane_line.detected = False
            right_lane_line.detected = False
            return left_lane_line, right_lane_line
        """

        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        left_lane_line.detected = True
        right_lane_line.detected = True

        #left_lane_line.diffs = left_lane_line.current_fit - left_fit
        left_lane_line.current_fit = left_fit
        left_lane_line.fits.append(left_fit)
        left_lane_line.allx = leftx
        left_lane_line.ally = lefty
        #right_lane_line.diffs = right_lane_line.current_fit - right_fit
        right_lane_line.current_fit = right_fit
        right_lane_line.fits.append(right_fit)
        right_lane_line.allx = rightx
        right_lane_line.ally = righty

        if left_lane_line.best_fit is None:
            left_lane_line.best_fit = left_fit
        if right_lane_line.best_fit is None:
            right_lane_line.best_fit = right_fit

    return left_lane_line, right_lane_line

def pct_change(val1, val2):
    diff = abs(val1 - val2)

    if diff == 0: return 0
    return (diff / val2) * 100.0

def fit_change_acceptable(current_fit, previous_fits):
    for i in range(len(current_fit)):
        j = 2
        #for j in range(len(current_fit[i])):
        std = []
        for k in range(len(previous_fits[i])):
            std.append(previous_fits[i][k][j])
        original_stddev = np.std(std)
        std.append(current_fit[i][j])
        new_stddev = np.std(std)

        print("Coeff", j, "oldstd", original_stddev, "new", new_stddev)
        if pct_change(original_stddev, new_stddev) > 50:
            return False
    return True

def topdown_crop(img):
    output = np.copy(img)

    mask_top = img.shape[0] / 2 - 50
    vertices_left = np.array([[
        (0,0),
        (img.shape[1] / 3, 0),
        (img.shape[1] / 3, img.shape[0]),
        (0, img.shape[0])]], dtype=np.int32)
    vertices_right = np.array([[
        (img.shape[1],0),
        (img.shape[1] - img.shape[1] / 4, 0),
        (img.shape[1] - img.shape[1] / 4, img.shape[0]),
        (img.shape[1], img.shape[0])]], dtype=np.int32)
    cv2.fillPoly(output, vertices_left, 0)
    cv2.fillPoly(output, vertices_right, 0)
    
    return output

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        self.fits = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

LEFT_LANE_LINE = Line()
RIGHT_LANE_LINE = Line()

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
clip1 = VideoFileClip("project_video.mp4").subclip(25, 30)
white_clip = clip1.fl_image(process_frame)
white_clip.write_videofile(white_output, audio=False)

