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



last_img = None
prev_pts = None

tracked_vehicles = []

framecount = 0
debounce = 0

def process_frame(img):
    global heatmaps, last_img, prev_pts, tracked_vehicles, framecount, debounce
    if last_img is None:
        last_img = img
        last_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(last_gray, maxCorners = 100, qualityLevel = 0.3, minDistance = 7)

    points, flow_lines = calculate_optical_flow(img, last_img, prev_pts)
    last_img = img
    prev_pts = points

    framecount += 1
    debounce += 1

    if (len(tracked_vehicles) < 1 and debounce > 5) or framecount > 15:
        last_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(last_gray, maxCorners = 100, qualityLevel = 0.3, minDistance = 7)
        find_new_vehicles(img, tracked_vehicles, flow_lines)
        framecount = 0
        debounce = 0

    i = 0
    for vehicle in tracked_vehicles[:]:
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
            if tracked_vehicles[i].missing_frames > 4:
                tracked_vehicles.remove(vehicle)
                debounce = 5
                find_new_vehicles(img, tracked_vehicles, flow_lines)
                i -= 1
        else:
            tracked_vehicles[i].missing_frames = 0
        i += 1

    for vehicle in tracked_vehicles:
        flow = get_flow_within_box(flow_lines, vehicle.bbox)
        vehicle.flow = flow
        draw_text(img, str(vehicle.probability), (vehicle.bbox[0][0], vehicle.bbox[0][1]))
        img = draw_optical_flow_lines(img, flow)

    draw_img = draw_labeled_bboxes(np.copy(img), tracked_vehicles)


    #cv2.imshow('image',draw_img)
    #cv2.waitKey(0)



    return draw_img


def draw_text(img, text, pos):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

output_file = 'output_dl_no_svm.mp4'
clip = VideoFileClip("project_video.mp4") #.subclip(35, 44)
processed_clip = clip.fl_image(process_frame)
processed_clip.write_videofile(output_file, audio=False)
