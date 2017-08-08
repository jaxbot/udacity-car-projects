import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Cropping2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from random import shuffle
import sklearn.utils
from sklearn.model_selection import train_test_split

EPOCHS = 1
TRAINING_DATA_DIR = "training_data/"
TRAINING_SETS = [
        "track_1_loop_1",            
        "track_1_loop_2",            
        "track_1_loop_3",            
        "track_1_loop_backwards",    
        "track_1_recovery_1",        
        "track_1_recovery_2",        
        "track_1_recovery_3",
        "track_2_loop_1",
        "track_2_loop_2",
        "udacity_sample_track_1"]        
STEERING_CORRECTION = 0.2

def get_local_image_path(training_set, filename):
    return TRAINING_DATA_DIR + training_set + "/IMG/" + filename.split("\\")[-1]

def load_image(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_training_data_csv(training_set):
    lines = []
    with open('training_data/' + training_set + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line[0] = get_local_image_path(training_set, line[0])
            line[1] = get_local_image_path(training_set, line[1])
            line[2] = get_local_image_path(training_set, line[2])
            lines.append(line)
    return lines

def load_all_training_sets():
    samples = []
    for training_set in TRAINING_SETS:
        samples += load_training_data_csv(training_set)

    return samples

def generator(samples, batch_size=64):
    num_samples = len(samples)

    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for line in batch_samples:
                try:
                    center_image = load_image(line[0])
                    left_image = load_image(line[1])
                    right_image = load_image(line[2])

                    steering_center = float(line[3])
                    steering_left = steering_center + STEERING_CORRECTION
                    steering_right = steering_center - STEERING_CORRECTION

                    images.append(center_image)
                    images.append(left_image)
                    images.append(right_image)
                    measurements.append(steering_center)
                    measurements.append(steering_left)
                    measurements.append(steering_right)

                    images.append(cv2.flip(center_image, 1))
                    images.append(cv2.flip(left_image, 1))
                    images.append(cv2.flip(right_image, 1))
                    measurements.append(-steering_center)
                    measurements.append(-steering_left)
                    measurements.append(-steering_right)
                except:
                    print("Failed to load", line[0], line[1], line[2])


            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

samples = load_all_training_sets()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

print("Total samples: ", len(samples))
"""
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

save_checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
        nb_val_samples=len(validation_samples), nb_epoch=EPOCHS, callbacks=[save_checkpoint])
"""
