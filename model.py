import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
import glob
import pickle
import time
import cv2

# Return HOG features using skimage.feature.hog
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                   visualise=vis, feature_vector=feature_vec)
    return features

def extract_features(imgs):
    i = 0
    features = []
    orientations = 11
    pix_per_cell = 16
    cell_per_block = 2

    for file in imgs:
        i += 1

        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        hog_features = []
        for channel in range(feature_image.shape[2]):
            feat = get_hog_features(feature_image[:,:,channel], 
                                orientations, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True)
            hog_features.append(feat)
        hog_features = np.ravel(hog_features)        

        # Append the new feature vector to the features list
        features.append(hog_features)

    # Return list of feature vectors
    return features

def train(car_images, negatives):
    # Extract car and not-car (negatives) HOG features
    notcar_features = extract_features(negatives)
    car_features = extract_features(car_images)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC
    svc = LinearSVC()
    svc.fit(X_train, y_train)

    return {
        'classifier': svc
    }

# Divide up into cars and notcars
car_images = glob.glob('training_data\\vehicles\\*.png')
negatives = glob.glob('training_data\\non-vehicles\\*.png')

print("Training with ", len(car_images), " car images and negatives ", len(negatives))

ret = train(car_images, negatives)
pickle.dump(ret, open("model.p", "wb"))
