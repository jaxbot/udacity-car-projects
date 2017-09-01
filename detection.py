import time
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
from moviepy.editor import VideoFileClip
import tensorflow as tf, sys
from scipy.ndimage.measurements import label

ystart = 300
ystop = 656
scale = 1.5
spatial_size = (8, 8)
hist_bins = 16

heatmaps = []
    
classifier_definition = pickle.load(open("model.p", "rb"))

svc = classifier_definition["classifier"]
X_scaler = classifier_definition["x_scaler"]
orient = classifier_definition["orientations"]
pix_per_cell = classifier_definition["pixels_per_cell"]
cell_per_block = classifier_definition["cell_per_block"]

label_lines = [line.rstrip() for line in tf.gfile.GFile("output_labels.txt")]

with tf.gfile.FastGFile("output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

sess = tf.Session()

def cnn_classify_subimage(subimg):
    global sess
    try:
        start = time.time()
        image_data = cv2.imencode('.jpg', subimg)[1].tostring()
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        end = time.time()
        print("Classifying window took ", end - start)

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        results = {}
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]

            print('%s (score = %.5f' % (human_string, score))
            results[human_string] = score

        return results['vehicles']
    except:
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
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2YUV) #convert_color(img_tosearch, conv='RGB2HLS')
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
            #subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

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
        cv2.rectangle(img, vehicle.bbox[0], vehicle.bbox[1], vehicle.color, 3)
    # Return the image
    return img

def average_heatmaps(heatmaps):
    print(len(heatmaps))
    if len(heatmaps) > 5:
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

def _bbox_collision(bbox1, bbox2):
    #return bbox1[0][0] >= bbox2[0][0] and bbox1[0][0] <= bbox2[1][0] and bbox1[0][1] >= bbox2[0][1] and bbox1[0][1] <= bbox2[1][1]
    return bbox1[0][0] < bbox2[1][0] and bbox1[1][0] > bbox2[0][0] and bbox1[0][1] < bbox2[1][1] and bbox1[1][1] > bbox2[0][1]

def bbox_collision(bbox1, bbox2):
    return _bbox_collision(bbox1, bbox2) or _bbox_collision(bbox2, bbox1)

def vehicle_intersects(tracked_vehicles, vehicle):
    for existing_vehicle in tracked_vehicles:
        if bbox_collision(vehicle.bbox, existing_vehicle.bbox):
            return True
    return False

def vehicle_is_subbox(tracked_vehicles, vehicle):
    for existing_vehicle in tracked_vehicles:
        bbox_e = existing_vehicle.bbox
        bbox_v = vehicle.bbox
        if bbox_v[0][0] > bbox_e[0][0] and bbox_v[1][0] < bbox_e[1][0] and bbox_v[0][1] > bbox_e[0][1] and bbox_v[1][1] < bbox_e[1][1]:
            return True, existing_vehicle
    return False, None


def find_new_vehicles(img, tracked_vehicles, flow_lines):
    boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)


    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, boxes)

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

        if not vehicle_intersects(tracked_vehicles, vehicle):
            tracked_vehicles.append(vehicle)

    """
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

        if not vehicle_intersects(tracked_vehicles, vehicle):
            tracked_vehicles.append(vehicle)
    """

def cnn_classify(img, labels):
    bboxes = []

    boxes_to_test = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        boxes_to_test.append(bbox)

    for bbox in boxes_to_test:
        startx = bbox[0][0]
        endx = bbox[1][0]
        starty = bbox[0][1]
        endy = bbox[1][1]

        height = endy - starty
        width = endx - startx

        if width < 64 or height < 64 or width > 800 or height > 600:
            continue

        padding = 1
        if endx * padding <= img.shape[1]:
            endx = int(endx * padding)
        else:
            endx = img.shape[1]

        img_section = img[starty:endy, startx:endx]
        overall_results = cnn_classify_subimage(img_section)
        if overall_results < 0.1:
            print("Overall results:", overall_results)
            continue
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

        vehicles = []
        x_shifts = []

        for x_shift in range(0, int(width - 299), 64):
            x_shifts.append(x_shift)
        x_shifts.append(int(width - 299))

        for x_shift in x_shifts:
            img_window = img_section[0:299, x_shift:299 + x_shift]

            results = cnn_classify_subimage(img_window)

            offset = (endx - startx) / width
            bbox = ((startx + int(x_shift * offset), starty), (startx + int((x_shift + 299) * offset), bbox[1][1]))
            found = False
            for vehicle in vehicles:
                if bbox_collision(vehicle.bbox, bbox):
                    if vehicle.probability < results:
                        vehicle.bbox = bbox
                        vehicle.probability = results
                        found = True
                        break
            if found == False and results > 0.8:
                vehicle = Vehicle()
                vehicle.bbox = bbox
                vehicle.probability = results
                vehicles.append(vehicle)
        for vehicle in vehicles:
            bboxes.append(vehicle.bbox)

    print("Roudn complete!")
    print(bboxes)
    return bboxes

COLORS = [(66, 134, 244),
        (66, 80, 244),
        (200, 66, 244),
        (244, 66, 104),
        (128, 244, 66),
        (244, 155, 66),
        (173, 244, 66),
        (66, 244, 167)]

current_color = 0

class Vehicle:
    bbox = []
    flow = []
    probability = 0
    missing_frames = 0
    color = (0, 0, 0)

    def __init__(self):
        global current_color
        self.color = COLORS[current_color]
        current_color += 1
        if current_color >= len(COLORS):
            current_color = 0

