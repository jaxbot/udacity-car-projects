import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from detection import *
import tensorflow as tf, sys
from os import listdir
from shutil import copyfile
import imutils
from imutils import contours
from skimage import measure
import time

# Global state shared between frames
last_img = None
prev_pts = None
tracked_vehicles = []
framecount = 0
debounce = 0

# Process a single frame based on state from previous iterations, if available.
def process_frame(img):
    global last_img, prev_pts, tracked_vehicles, framecount, debounce

    # Create optical flow points if none exist
    if last_img is None:
        last_img = img
        last_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(last_gray, maxCorners = 100, qualityLevel = 0.3, minDistance = 7)

    points, flow_lines = calculate_optical_flow(img, last_img, prev_pts)
    last_img = img
    prev_pts = points

    framecount += 1
    debounce += 1

    # If no vehicles have been found and we haven't checked in 5 frames, or every 15 frames regardless, look for vehicles.
    if (len(tracked_vehicles) < 1 and debounce > 5) or framecount > 15:
        last_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(last_gray, maxCorners = 100, qualityLevel = 0.3, minDistance = 7)
        find_new_vehicles(img, tracked_vehicles, flow_lines)
        framecount = 0
        debounce = 0

    # Iterate through every vehicle we are currently tracking, if any
    i = 0
    tracked_vehicles_copy = tracked_vehicles[:]
    for vehicle in tracked_vehicles_copy:
        # Move the bounding box based on optical flow within its bounding box
        movement = calculate_box_movement(vehicle.flow)
        new_box = ((vehicle.bbox[0][0] + movement[0], vehicle.bbox[0][1] + movement[1]),
                (vehicle.bbox[1][0] + movement[0], vehicle.bbox[1][1] + movement[1]))

        vehicle.bbox = new_box

        # Check if the vehicle is still in the box we predict it to be in
        vehicle_subimage = img[new_box[0][1]:new_box[1][1], new_box[0][0]:new_box[1][0]]
        results = cnn_classify_subimage(vehicle_subimage)
        vehicle.probability = results

        print("The vehicle we expect to be has score ", results)

        if results < 0.8:
            tracked_vehicles[i].missing_frames += 1
            # If the vehicle does not exist for 5 frames, remove it and check for new vehicles next frame
            if tracked_vehicles[i].missing_frames > 4:
                tracked_vehicles.remove(vehicle)
                debounce = 5
                find_new_vehicles(img, tracked_vehicles, flow_lines)
                i -= 1
        else:
            tracked_vehicles[i].missing_frames = 0
        i += 1

    # Draw tracked vehicles that still exist
    for vehicle in tracked_vehicles:
        flow = get_flow_within_box(flow_lines, vehicle.bbox)
        vehicle.flow = flow
        draw_text(img, str(vehicle.probability), (vehicle.bbox[0][0], vehicle.bbox[0][1]))
        img = draw_optical_flow_lines(img, flow)

    draw_img = draw_vehicles(np.copy(img), tracked_vehicles)

    return draw_img

# Draws text on an imagine at a specified location
def draw_text(img, text, pos):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Load video and kick off pipeline
output_file = 'tmp.mp4'
clip = VideoFileClip("project_video.mp4").subclip(28, 30)
processed_clip = clip.fl_image(process_frame)
processed_clip.write_videofile(output_file, audio=False)
