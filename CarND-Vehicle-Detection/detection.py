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

classifier_definition = pickle.load(open("model.p", "rb"))

SVC = classifier_definition["classifier"]
ORIENTATIONS = 11
PIX_PER_CELL = 16
CELL_PER_BLOCK = 2
Y_START = 300
Y_STOP = 656
SCALE = 1.5

OPTICAL_FLOW_COLORS = np.random.randint(0,255,(100,3))

# Load TF image classifier
label_lines = [line.rstrip() for line in tf.gfile.GFile("output_labels.txt")]
with tf.gfile.FastGFile("output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Use a global session as starting one is expensive
sess = tf.Session()

# Classify a single ~299x299 image with CNN and return the probability of it being a vehicle
def cnn_classify_subimage(subimg):
    global sess
    try:
        # TODO: Use the pixel data tensor to avoid having to jpeg encode features.
        image_data = cv2.imencode('.jpg', subimg)[1].tostring()
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        top_predictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

        results = {}
        for node_id in top_predictions:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]

            print('%s (score = %.5f' % (human_string, score))
            results[human_string] = score

        return results['vehicles']
    except:
        return 0

def get_hog_features(img, orient, pix_per_cell, cell_per_block, feature_vec=True):
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                   visualise=False, feature_vector=feature_vec)
    return features

# Extract features using hog sub-sampling and return potential bounding boxes
# Taken from Udacity CarND Vehicle Tracking module
def find_cars(img, ystart, ystop, scale, svc, orient, pix_per_cell, cell_per_block):
    boxes = []
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2YUV)
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
    return heatmap

def draw_labeled_bboxes(img, bboxes):
    # Iterate through all detected cars
    for bbox in bboxes:
        cv2.rectangle(img, bbox[0], bbox[1], COLORS[1], 3)
    # Return the image
    return img

def draw_vehicles(img, vehicles):
    # Iterate through all detected cars
    for vehicle in vehicles:
        cv2.rectangle(img, vehicle.bbox[0], vehicle.bbox[1], vehicle.color, 3)
    # Return the image
    return img

# Calculate the optical flow between two frames given previous tracked points
def calculate_optical_flow(img, last_img, prev_pts):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    last_img = cv2.cvtColor(last_img, cv2.COLOR_BGR2GRAY)

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(last_img, gray_img, prev_pts, None)

    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]
    flow_lines = (good_new,good_old)

    return next_pts, flow_lines

# Draw optical flow lines on an image
def draw_optical_flow_lines(img, flow_lines):
    global OPTICAL_FLOW_COLORS
    mask = np.zeros_like(img)

    i = 0
    for (new,old) in flow_lines:
        x,y = new.ravel()
        u,v = old.ravel()
        mask = cv2.line(mask, (x,y),(u,v), OPTICAL_FLOW_COLORS[i].tolist(), 2)
        frame = cv2.circle(img, (x, y), 5, OPTICAL_FLOW_COLORS[i].tolist(), -1)
        i = i + 1
    img = cv2.add(img, mask)

    return img

# Find flow lines and tracked points that exist within a bounding box
def get_flow_within_box(flow_lines, bbox):
    retval = []

    if len(flow_lines) < 2:
        return retval

    for i,(new,old) in enumerate(zip(flow_lines[0], flow_lines[1])):
        x,y = new.ravel()

        if x >= bbox[0][0] and x <= bbox[1][0] and y >= bbox[0][1] and y <= bbox[1][1]:
            # Flow is within box
            retval.append((new, old))

    return retval

# Calculate the movement between a set of flow lines
def calculate_box_movement(flow_lines):
    avg_x = 0
    avg_y = 0
    count = 0

    for (new,old) in flow_lines:
        x1,y1 = new.ravel()
        x2,y2 = old.ravel()

        x_mov = x1 - x2
        y_mov = y1 - y2

        # Ignore major shifts between frames caused by noisy tracking points
        if abs(x_mov) > 5 or abs(y_mov) > 5:
            print("Ignoring: X:", x_mov, "Y:", y_mov)
        else:
            avg_x += x_mov
            avg_y += y_mov
            count += 1
    if count > 0:
        return (int((avg_x / count)), int((avg_y / count)))
    return (0, 0)

# Returns whether the two bounding boxes collide
def _bbox_collision(bbox1, bbox2):
    return bbox1[0][0] < bbox2[1][0] and bbox1[1][0] > bbox2[0][0] and bbox1[0][1] < bbox2[1][1] and bbox1[1][1] > bbox2[0][1]

# Returns whether the two bounding boxes collide
def bbox_collision(bbox1, bbox2):
    return _bbox_collision(bbox1, bbox2) or _bbox_collision(bbox2, bbox1)

# Returns whether a vehicle's bounding box collides with existing tracked vehicles
def vehicle_intersects(tracked_vehicles, vehicle):
    for existing_vehicle in tracked_vehicles:
        if bbox_collision(vehicle.bbox, existing_vehicle.bbox):
            return True
    return False

# Finds new vehicles in an image
def find_new_vehicles(img, tracked_vehicles, flow_lines):
    # Find potential matches from HOG classifier
    boxes = find_cars(img,
            Y_START,
            Y_STOP,
            SCALE,
            SVC,
            ORIENTATIONS,
            PIX_PER_CELL,
            CELL_PER_BLOCK)

    # Create a heatmap from HOG classifier
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    heat = add_heat(heat, boxes)

    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    # Narrow down boxes using pixel CNN classifier
    final_boxes = cnn_classify(img, labels)

    # Add any new vehicles that do not intersect with existing ones
    for box in final_boxes:
        vehicle = Vehicle()
        vehicle.bbox = box
        flow = get_flow_within_box(flow_lines, box)
        vehicle.flow = flow

        if not vehicle_intersects(tracked_vehicles, vehicle):
            tracked_vehicles.append(vehicle)

# Classify vehicles within a heatmap
def cnn_classify(img, labels):
    bboxes = []
    boxes_to_test = []

    # Create a list of bounding boxes from the input heatmap
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        boxes_to_test.append(bbox)

    # Test each bounding box for vehicles
    for bbox in boxes_to_test:
        startx = bbox[0][0]
        endx = bbox[1][0]
        starty = bbox[0][1]
        endy = bbox[1][1]

        height = endy - starty
        width = endx - startx

        # Ignore overly large or too small sections
        if width < 64 or height < 64 or width > 800 or height > 600:
            continue

        img_section = img[starty:endy, startx:endx]
        # Run a classification on the entire bounding box, and discard if there's a low chance of
        # vehicles being found within it.
        overall_results = cnn_classify_subimage(img_section)
        if overall_results < 0.1:
            print("Overall results:", overall_results)
            continue

        width = img_section.shape[1]
        height = img_section.shape[0]

        # Create an image with 299 as the minimal dimension
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

        # Generate a list of window shifts to iterate through (just makes the math easier).
        for x_shift in range(0, int(width - 299), 64):
            x_shifts.append(x_shift)
        x_shifts.append(int(width - 299))

        # Sweep the image horizontally looking for vehicles in 299x299 squares
        # This helps find cases where two vehicles are side by side and included in a single heatmap label
        for x_shift in x_shifts:
            img_window = img_section[0:299, x_shift:299 + x_shift]

            results = cnn_classify_subimage(img_window)

            offset = (endx - startx) / width
            bbox = ((startx + int(x_shift * offset), starty), (startx + int((x_shift + 299) * offset), bbox[1][1]))
            found = False
            # If this bounding box collides with others, only accept it if its probability is higher
            for vehicle in vehicles:
                if bbox_collision(vehicle.bbox, bbox):
                    if vehicle.probability < results:
                        vehicle.bbox = bbox
                        vehicle.probability = results
                        found = True
                        break
            # If it does not collide and it meets the minimum threshold, accept it
            if found == False and results > 0.8:
                vehicle = Vehicle()
                vehicle.bbox = bbox
                vehicle.probability = results
                vehicles.append(vehicle)
        for vehicle in vehicles:
            bboxes.append(vehicle.bbox)

    return bboxes

# Bounding box colors
COLORS = [(66, 134, 244),
        (66, 80, 244),
        (200, 66, 244),
        (244, 66, 104),
        (128, 244, 66),
        (244, 155, 66),
        (173, 244, 66),
        (66, 244, 167)]
current_color = 0

# Class of a tracked vehicle bounding box.
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
