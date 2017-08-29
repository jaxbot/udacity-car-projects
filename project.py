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

        cropped_box = img[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]

        height = cropped_box.shape[0]
        width = cropped_box.shape[1]

        if width < 64 or height < 64:
            continue

        splits = 1
        if width / height >= 2:
            splits = int(width / height)

        if width > height:
            width = width / height * 299
            height = 299
        else:
            height = height / width * 299
            width = 299

        dimensions = (int(width), int(height))
        cropped_img = cv2.resize(cropped_box, dimensions, interpolation = cv2.INTER_AREA)

        for split in range(splits):
            startx = split * 299
            if width - startx < 299:
                startx = width - 299
            img_section = cropped_img[0:299, startx:startx + 299]
            hls = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HLS)
            sobelx = cv2.Sobel(hls[:,:,2], cv2.CV_64F, 1, 0, ksize=9)
            abs_sobelx = np.absolute(sobelx)
            scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
            blurred = cv2.GaussianBlur(scaled_sobel, (11, 11), 0)
            thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=4)

            #cv2.imshow('image',thresh)
            #cv2.waitKey(0)
            clabels = measure.label(thresh, neighbors=8, background=0)
            mask = np.zeros(thresh.shape, dtype="uint8")
             
            # loop over the unique components
            for lbl in np.unique(clabels):
                    # if this is the background label, ignore it
                    if lbl == 0:
                            continue
             
                    # otherwise, construct the label mask and count the
                    # number of pixels 
                    labelMask = np.zeros(thresh.shape, dtype="uint8")
                    labelMask[clabels == lbl] = 255
                    numPixels = cv2.countNonZero(labelMask)
             
                    # if the number of pixels in the component is sufficiently
                    # large, then add it to our mask of "large blobs"
                    if numPixels > 100:
                            mask = cv2.add(mask, labelMask)

            #cv2.imshow('image',mask)
            #cv2.waitKey(0)
            # find the contours in the mask, then sort them from left to
            # right
            im2, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
             
            # loop over the contours
            for (i, c) in enumerate(cnts):
                    print("Contour loop!")
                    # draw the bright spot on the image
                    (x, y, w, h) = cv2.boundingRect(c)
                    print(x, y, w, h)
                    print("drawing!")
                    cv2.rectangle(cropped_img, (x, y), (x + w, y + h), (0,0,255), 6)
                    cv2.putText(cropped_img, "#{}".format(i + 1), (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            #cv2.imshow('image',cropped_img)
            #cv2.waitKey(0)

            image_data = cv2.imencode('.jpg', img_section)[1].tostring()
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            results = {}
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]

                print('%s (score = %.5f' % (human_string, score))
                results[human_string] = score

            if results['vehicles'] > 0.8:
                offset = int(startx * (cropped_box.shape[1] / width))
                print("OFFSET: ", offset)
                bboxes.append(((bbox[0][0] + offset, bbox[0][1]), (bbox[0][0] + offset + 299, bbox[1][1])))
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
    heat = average_heatmaps(heatmaps)

    if len(tracked_vehicles) < 1 or framecount > 5:
        framecount = 0
        tracked_vehicles = []
        last_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(last_gray, maxCorners = 100, qualityLevel = 0.3, minDistance = 7)

        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        final_boxes = cnn_classify(img, labels)

        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = add_heat(heat, final_boxes)
        heatmaps.append(heat)
        heat = average_heatmaps(heatmaps)
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
            tracked_vehicles.append(vehicle)
            img = draw_optical_flow_lines(img, flow)
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


    #cv2.imshow('image',img)
    #cv2.waitKey(0)



    return draw_img


class Vehicle:
    bbox = []
    flow = []

output_file = 'output_dl_no_svm.mp4'
clip = VideoFileClip("project_video.mp4").subclip(18, 30)
processed_clip = clip.fl_image(process_frame)
processed_clip.write_videofile(output_file, audio=False)
