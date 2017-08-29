import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip
import tensorflow as tf, sys
from scipy.ndimage.measurements import label

label_lines = [line.rstrip() for line in tf.gfile.GFile("output_labels.txt")]

with tf.gfile.FastGFile("output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def cnn_classify_subimage(subimg):
    with tf.Session() as sess:
        image_data = cv2.imencode('.jpg', subimg)[1].tostring()
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
            return 1
    return 0

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    boxes = []
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = img_tosearch #convert_color(img_tosearch, conv='RGB2HLS')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            #test_prediction = cnn_classify_subimage(subimg)

            # Get color features
            #spatial_features = bin_spatial(subimg, size=spatial_size)
            #hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(hog_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                
    return boxes

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # normalize
    max_box = np.max(heatmap)

    for i in range(len(heatmap)):
        heatmap[i] = heatmap[i] / max_box

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, vehicles):
    # Iterate through all detected cars
    for vehicle in vehicles:
        cv2.rectangle(img, vehicle.bbox[0], vehicle.bbox[1], (0,0,255), 6)
    # Return the image
    return img

def average_heatmaps(heatmaps):
    print(len(heatmaps))
    if len(heatmaps) > 10:
        heatmaps.pop(0)

    average_heatmap = heatmaps[0]
    if len(heatmaps) > 0:
        for i in range(1, len(heatmaps)):
            average_heatmap = np.logical_and(heatmaps[i], average_heatmap)
            print("averaging")

    return average_heatmap

color = np.random.randint(0,255,(100,3))
def calculate_optical_flow(img, last_img, prev_pts):
    global color
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    last_img = cv2.cvtColor(last_img, cv2.COLOR_BGR2GRAY)

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(last_img, gray_img, prev_pts, None)

    # Select good points
    good_new = next_pts[status==1]
    good_old = prev_pts[status==1]
    flow_lines = (good_new,good_old)

    return next_pts, flow_lines

def draw_optical_flow_lines(img, flow_lines):
    global color
    mask = np.zeros_like(img)

    i = 0
    for (new,old) in flow_lines:
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(img,(a,b),5,color[i].tolist(),-1)
        i = i + 1
    img = cv2.add(img,mask)

    return img

def get_flow_within_box(flow_lines, bbox):
    retval = []

    if len(flow_lines) < 2:
        return retval

    for i,(new,old) in enumerate(zip(flow_lines[0], flow_lines[1])):
        a,b = new.ravel()
        c,d = old.ravel()

        if a >= bbox[0][0] and a <= bbox[1][0] and b >= bbox[0][1] and b <= bbox[1][1]:
            # Flow is within box
            retval.append((new, old))

    return retval

def calculate_box_movement(flow_lines):
    avg_x = 0
    avg_y = 0
    count = 0
    for (new,old) in flow_lines:
        a,b = new.ravel()
        c,d = old.ravel()

        x_mov = a - c
        y_mov = b - d

        if abs(x_mov) > 5 or abs(y_mov) > 5:
            print("Ignoring: X:", x_mov, "Y:", y_mov)
        else:
            avg_x += x_mov
            avg_y += y_mov
            count += 1
    if count > 0:
        return (int((avg_x / count)), int((avg_y / count)))
    return (0, 0)
