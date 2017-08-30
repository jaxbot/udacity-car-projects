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

    
ystart = 400
ystop = 656
scale = 1.5
    
classifier_definition = pickle.load(open("model.p", "rb"))

svc = classifier_definition["classifier"]
X_scaler = classifier_definition["x_scaler"]
orient = classifier_definition["orientations"]
pix_per_cell = classifier_definition["pixels_per_cell"]
cell_per_block = classifier_definition["cell_per_block"]

spatial_size = (8, 8)
hist_bins = 16

heatmaps = []

sess = tf.Session()

def cnn_classify(img, labels):
    global sess
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image

        startx = bbox[0][0]
        endx = bbox[1][0]
        starty = bbox[0][1]
        endy = bbox[1][1]

        height = endy - starty
        width = endx - startx

        if width < 64 or height < 64 or width > 800 or height > 600:
            continue

        padding = 1.5
        if endy * padding <= img.shape[0]:
            endy = int(endy * padding)
        else:
            endy = img.shape[0]
        if endx * padding <= img.shape[1]:
            endx = int(endx * padding)
        else:
            endx = img.shape[1]

        img_section = img[starty:endy, startx:endx]
        #cv2.imshow('image',img_section)
        #cv2.waitKey(0)

        width = img_section.shape[1]
        height = img_section.shape[0]

        if width > height:
            width = width / height * 299
            height = 299
        else:
            height = height / width * 299
            width = 299

        dimensions = (int(width), int(height))
        img_section = cv2.resize(img_section, dimensions, interpolation = cv2.INTER_AREA)

        highest_result = 0
        best_box = None
        x_shifts = []

        for x_shift in range(0, int(width - 299), 64):
            x_shifts.append(x_shift)
        x_shifts.append(int(width - 299))

        for x_shift in x_shifts:
            img_window = img_section[0:299, x_shift:299 + x_shift]

            #cv2.imshow('image',img_window)
            #cv2.waitKey(0)

            image_data = cv2.imencode('.jpg', img_window)[1].tostring()
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            results = {}
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]

                print('%s (score = %.5f' % (human_string, score))
                results[human_string] = score

            if results['vehicles'] > highest_result:
                highest_result = results['vehicles']
                offset = width / (endx - startx)
                #cv2.imshow('image',img_section)
                #cv2.waitKey(0)
                best_box = ((startx + int(x_shift * offset), starty), (startx + int((x_shift + 299) * offset), bbox[1][1]))
        if highest_result > 0.8:
            bboxes.append(best_box)

    print("Roudn complete!")
    print(bboxes)
    return bboxes


last_img = None
prev_pts = None

tracked_vehicles = []

framecount = 0

def process_frame(img):
    global heatmaps, last_img, prev_pts, tracked_vehicles, framecount
    if last_img is None:
        last_img = img
        last_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(last_gray, maxCorners = 100, qualityLevel = 0.3, minDistance = 7)

    points, flow_lines = calculate_optical_flow(img, last_img, prev_pts)
    last_img = img
    prev_pts = points

    framecount += 1

    boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, boxes)
    heatmaps.append(heat)
    #heat = average_heatmaps(heatmaps)

    if len(tracked_vehicles) < 1 or framecount > 15:
        last_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(last_gray, maxCorners = 100, qualityLevel = 0.3, minDistance = 7)

        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        final_boxes = cnn_classify(img, labels)

        for box in final_boxes:
            vehicle = Vehicle()
            vehicle.bbox = box
            flow = get_flow_within_box(flow_lines, box)
            vehicle.flow = flow

            print("Flow!", flow)
            if framecount > 15:
                framecount = 0
                tracked_vehicles = []
            tracked_vehicles.append(vehicle)
            img = draw_optical_flow_lines(img, flow)

        """
        if len(final_boxes) > 0:
            heat = np.zeros_like(img[:,:,0]).astype(np.float)
            heat = add_heat(heat, final_boxes)
            heatmap = np.clip(heat, 0, 255)
            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            for car_number in range(1, labels[1]+1):
                # Find pixels with each car_number label value
                nonzero = (labels[0] == car_number).nonzero()
                # Identify x and y values of those pixels
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                # Define a bounding box based on min/max x and y
                box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

                vehicle = Vehicle()
                vehicle.bbox = box
                flow = get_flow_within_box(flow_lines, box)
                vehicle.flow = flow

                print("Flow!", flow)
                if framecount > 15:
                    framecount = 0
                    tracked_vehicles = []
                tracked_vehicles.append(vehicle)
                img = draw_optical_flow_lines(img, flow)
        """
        draw_img = draw_labeled_bboxes(np.copy(img), tracked_vehicles)
    else:
        for vehicle in tracked_vehicles:
            movement = calculate_box_movement(vehicle.flow)
            new_box = ((vehicle.bbox[0][0] + movement[0], vehicle.bbox[0][1] + movement[1]),
                    (vehicle.bbox[1][0] + movement[0], vehicle.bbox[1][1] + movement[1]))

            vehicle.bbox = new_box
            flow = get_flow_within_box(flow_lines, new_box)
            vehicle.flow = flow
            img = draw_optical_flow_lines(img, flow)

        draw_img = draw_labeled_bboxes(np.copy(img), tracked_vehicles)


    #cv2.imshow('image',draw_img)
    #cv2.waitKey(0)



    return draw_img


class Vehicle:
    bbox = []
    flow = []

output_file = 'output_dl_no_svm.mp4'
clip = VideoFileClip("project_video.mp4").subclip(9, 12)
processed_clip = clip.fl_image(process_frame)
processed_clip.write_videofile(output_file, audio=False)
