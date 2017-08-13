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

def process_frame(frame):
    img_size = (frame.shape[1], frame.shape[0])
    ROAD_TRANSFORM_DEST = np.float32([
        [img_size[0] - 490, 0],
        [img_size[0] - 490, img_size[1]],
        [490, img_size[1]],
        [490, 0]])
    warp_matrix = cv2.getPerspectiveTransform(ROAD_TRANSFORM_SRC, ROAD_TRANSFORM_DEST)
    warped_image = cv2.warpPerspective(frame, warp_matrix, img_size, flags=cv2.INTER_LINEAR)

    thresholded = thresholded_saturation(warped_image, threshold=(60,255))
    window_centroids = find_window_centroids(thresholded, window_width, window_height, margin)
    output = draw_centroids(thresholded, window_width, window_height, window_centroids)

    # Unwarp
    warp_matrix = cv2.getPerspectiveTransform(ROAD_TRANSFORM_DEST, ROAD_TRANSFORM_SRC)
    warped_output = cv2.warpPerspective(output, warp_matrix, img_size, flags=cv2.INTER_LINEAR)
    return warped_output

def thresholded_saturation(img, threshold=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    saturation = hls[:,:,2]
    binary_output = np.zeros_like(saturation)
    binary_output[(saturation > threshold[0]) & (saturation <= threshold[1])] = 255

    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def draw_centroids(img, window_width, window_height, window_centroids):
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows     
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,img,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,img,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((img,img,img)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
     
    # If no window centers found, just display orginal road image
    else:
        print("Just showing original")
        output = np.array(cv2.merge((img,img,img)),np.uint8)
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
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_frame)
white_clip.write_videofile(white_output, audio=False)

