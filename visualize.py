import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.callbacks import TensorBoard
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Cropping2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from random import shuffle
import sklearn.utils
from sklearn.model_selection import train_test_split

model = load_model('model.h5')
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
        write_graph=True, write_images=True)
tensorboard.set_model(model)
