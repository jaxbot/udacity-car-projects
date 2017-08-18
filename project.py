import cv2
import numpy as np
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt


VERBOSE = False
WINDOW_SIZE = 6
ROAD_TRANSFORM_SRC = np.float32([
        [684, 450],
        [1030, 677],
        [280, 677],
        [598, 450]])
# Meters per pixel in y dimension
YM_PER_PIX = 30 / 720
# Meters per pixel in x dimension
XM_PER_PIX = 3.7 / 700

# Calibrate camera using a chessboard image and the given board size.
def calibrate_camera(chess_img, board_size=(8,6)):
    grayscale = grayscale_img(chess_img)
    return_val, corners = cv2.findChessboardCorners(grayscale, board_size, None)

    assert return_val != 0, "Could not find corners."

    image_points = corners
    object_points = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    object_points[:,:2] = np.mgrid[0:board_size[0],0:board_size[1]].T.reshape(-1, 2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([object_points], [image_points], grayscale.shape[::-1], None, None)

    return mtx, dist

# Convert a BGR image to grayscale
def grayscale_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Undistort an image given camera calibration mtx and dist
def undistort_img(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

def process_frame(frame):
    global LEFT_LANE_LINE, RIGHT_LANE_LINE, CAMERA_MTX, CAMERA_DIST, VERBOSE

    # Main frame processing pipeline
    img_size = (frame.shape[1], frame.shape[0])

    # Undistort frame with camera calibration constants
    frame = undistort_img(frame, CAMERA_MTX, CAMERA_DIST)

    # Transform frame into top-down view
    road_transform_dest = np.float32([
        [img_size[0] - 490, 0],
        [img_size[0] - 490, img_size[1]],
        [490, img_size[1]],
        [490, 0]])
    warp_matrix = cv2.getPerspectiveTransform(ROAD_TRANSFORM_SRC, road_transform_dest)
    warped_image = cv2.warpPerspective(frame, warp_matrix, img_size, flags=cv2.INTER_LINEAR)

    # Run threshold pipeline to create binary image of possible lane lines
    thresholded = thresholded_all(warped_image)
    # Crop left and right sides of top-down image to reduce noise and remove other lane markers from image
    thresholded = topdown_crop(thresholded)

    # Calculate lane lines for frame
    LEFT_LANE_LINE, RIGHT_LANE_LINE = find_lane_lines(thresholded, LEFT_LANE_LINE, RIGHT_LANE_LINE)

    # Average together line coefficients from past WINDOW_SIZE frames
    average_lines([LEFT_LANE_LINE, RIGHT_LANE_LINE])

    # Draw polygon from lane lines
    color_warp = draw_lines(LEFT_LANE_LINE, RIGHT_LANE_LINE, thresholded)

    # Unwarp lane image into real-world space
    warp_matrix = cv2.getPerspectiveTransform(road_transform_dest, ROAD_TRANSFORM_SRC)
    color_unwarp = cv2.warpPerspective(color_warp, warp_matrix, img_size, flags=cv2.INTER_LINEAR)

    # Overlay lane drawing onto the original frame
    result = cv2.addWeighted(frame, 1, color_unwarp, 0.3, 0)

    # Print frame stats onto video
    curve = measure_curvature(LEFT_LANE_LINE, RIGHT_LANE_LINE, thresholded)
    vehicle_offset = calculate_vehicle_position(LEFT_LANE_LINE, RIGHT_LANE_LINE, thresholded)

    draw_text(result, "CurveR: " + str(round(curve)) + "m", (50, result.shape[0] - 50))
    draw_text(result, "Offset: " + str(round(vehicle_offset, 2)) + "m", (50, result.shape[0] - 80))

    # Show the frame's transformations frame-by-frame when debugging
    if VERBOSE:
        cv2.imshow('thresholded', thresholded)
        cv2.imshow('img_warped', warped_image)
        cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result

# Average the past WINDOW_SIZE line fits together and set best_fit appropriately.
def average_lines(lines):
    global WINDOW_SIZE

    for line in lines:
        best_fit = (0, 0, 0)
        for fit in line.fits:
            best_fit = (
                    best_fit[0] + fit[0],
                    best_fit[1] + fit[1],
                    best_fit[2] + fit[2])

        best_fit = (best_fit[0] / len(line.fits),
                best_fit[1] / len(line.fits),
                best_fit[2] / len(line.fits))
        line.best_fit = best_fit

        # Remove oldest buffer entry as buffer fills
        if len(line.fits) > WINDOW_SIZE and line.detected:
            line.fits.pop(0)

def draw_lines(left_lane, right_lane, img):
    # Code taken from CarND module
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
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    return color_warp

def measure_curvature(left_lane_line, right_lane_line, img):
    global YM_PER_PIX, XM_PER_PIX

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )

    y_eval = np.max(ploty)

    # Create new polynomial fits based on raw pixel data converted to meters
    left_fit = np.polyfit(left_lane_line.ally*YM_PER_PIX, left_lane_line.allx*XM_PER_PIX, 2)
    right_fit = np.polyfit(right_lane_line.ally*YM_PER_PIX, right_lane_line.allx*XM_PER_PIX, 2)

    # Calculate the left and right lane radii of curvature per the formula from
    # http://www.intmath.com/applications-differentiation/8-radius-curvature.php
    left_radius = (
            (1 + (2 * left_fit[0] * y_eval * YM_PER_PIX + left_fit[1])**2)**1.5) /\
            np.absolute(2 * left_fit[0])
    right_radius = (
            (1 + (2* right_fit[0] *y_eval * YM_PER_PIX + right_fit[1])**2)**1.5) /\
            np.absolute(2 * right_fit[0])

    return (left_radius + right_radius) / 2

def calculate_vehicle_position(left_lane_line, right_lane_line, img):
    global XM_PER_PIX
    lane_center = (left_lane_line.best_fit[2] + right_lane_line.best_fit[2]) / 2
    image_center = img.shape[1] / 2

    # Distance from center is the image center minus the midpoint of lane positions
    distance = lane_center - image_center

    return distance * XM_PER_PIX

def thresholded_all(img, s_threshold=(30, 255),
        l_threshold=(65, 255), sobel_threshold = (2, 255)):
    # Apply all thresholds in pipeline to frame
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    saturation = hls[:,:,2]
    lightness = hls[:,:,1]

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    # Sobel must be normalized as some gradients can be negative
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Create binary image where white pixels represent pixels that meet all filter thresholds
    binary_output = np.zeros_like(saturation)
    binary_output[
            (lightness > l_threshold[0]) & (lightness <= l_threshold[1]) &
            (saturation > s_threshold[0]) & (saturation <= s_threshold[1]) &
            (scaled_sobel > sobel_threshold[0]) & (scaled_sobel <= sobel_threshold[1])] = 255

    return binary_output

def find_lane_lines(img, left_lane_line, right_lane_line):
    # Code taken from CarND module
    # Find lane lines using histogram method

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows to the height of the image divided by the number of windows
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
        # If you found > minpix pixels, recenter next window on their mean position to the found pixel cloud
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Get non-zero pixel positions from lane indices
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # If any pixel lists are empty, we can not conclude anything from this frame and will discard
    # both lane lines
    if len(leftx) < 1 or len(lefty) < 1 or len(rightx) < 1 or len(righty) < 1:
        left_lane_line.detected = False
        right_lane_line.detected = False

        print("Could not detect lane!")
    else:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # If we have a previous lane line fit, check if this one
        # is acceptable.
        if len(left_lane_line.fits) > 0:
            lanes = [left_lane_line, right_lane_line]
            fits = [left_fit, right_fit]
            # If the distance between the found lines is too high or low, reject this frame
            # This prevents our average lane from being polluted with outliers from lanes where
            # noise was detected as lane lines
            if abs(left_fit[2] - right_fit[2]) < 250 or abs(left_fit[2] - right_fit[2]) > 550:
                print("Debug: Discarding frame!")
                left_lane_line.detected = False
                right_lane_line.detected = False
                return left_lane_line, right_lane_line

        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Generate a polygon to illustrate the search window area.
        # This code is taken from the CarND CV module.
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Store the fit in the given lane line objects
        left_lane_line.detected = True
        right_lane_line.detected = True

        left_lane_line.current_fit = left_fit
        left_lane_line.fits.append(left_fit)
        left_lane_line.allx = leftx
        left_lane_line.ally = lefty

        right_lane_line.current_fit = right_fit
        right_lane_line.fits.append(right_fit)
        right_lane_line.allx = rightx
        right_lane_line.ally = righty

        if left_lane_line.best_fit is None:
            left_lane_line.best_fit = left_fit
        if right_lane_line.best_fit is None:
            right_lane_line.best_fit = right_fit

    return left_lane_line, right_lane_line

def topdown_crop(img):
    # Crop out the left and right sides of the input image
    output = np.copy(img)

    mask_top = img.shape[0] / 2 - 50
    vertices_left = np.array([[
        (0,0),
        (img.shape[1] / 3.5, 0),
        (img.shape[1] / 3.5, img.shape[0]),
        (0, img.shape[0])]], dtype=np.int32)
    vertices_right = np.array([[
        (img.shape[1],0),
        (img.shape[1] - img.shape[1] / 3.5, 0),
        (img.shape[1] - img.shape[1] / 3.5, img.shape[0]),
        (img.shape[1], img.shape[0])]], dtype=np.int32)
    # Drawing a black polygon over the image is an efficient way of masking without having to
    # iterate through the matrix in python bytecode.
    cv2.fillPoly(output, vertices_left, 0)
    cv2.fillPoly(output, vertices_right, 0)

    return output

def draw_text(img, text, pos):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

class Line():
    def __init__(self):
        # Whether or not the lane line was detected in the past iteration
        self.detected = False
        # Array of past N fit coefficients
        self.fits = []
        # Window average fit coefficients
        self.best_fit = None
        # Current fit coefficients
        self.current_fit = None
        # Detected lane pixels X
        self.allx = None
        # Detected lane pixels Y
        self.ally = None

# Calibrate camera used for input frames
img = cv2.imread("camera_cal/calibration2.jpg")
CAMERA_MTX, CAMERA_DIST = calibrate_camera(img)

LEFT_LANE_LINE = Line()
RIGHT_LANE_LINE = Line()

# Run pipeline on project video
output_file = 'output.mp4'
clip = VideoFileClip("project_video.mp4")
processed_clip = clip.fl_image(process_frame)
processed_clip.write_videofile(output_file, audio=False)
