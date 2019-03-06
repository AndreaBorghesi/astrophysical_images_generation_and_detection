'''
Andrea Borghesi
University of Bologna
    2019-03-02

Script to train the NN using DAVIDE 
'''

#!/usr/bin/python

import numpy as np
import sys
import os
import subprocess
#import pickle
#from pathlib import Path
import math
import time
import pandas as pd 
#from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import tensorflow as tf
from keras.models import load_model, Sequential, Model
from keras import backend as K 
from keras import optimizers, initializers, regularizers
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Input
from keras.layers import UpSampling1D, Lambda, Dropout, merge
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, History

_batch_size = 32
_epochs = 10

base_dir = '/davide/home/userexternal/aborghes/'
base_dir += 'astrophysical_images_generation_and_detection/'
img_dir_train = base_dir + 'images_set/'
img_dir_train_small = base_dir + 'images_set_small/'
img_dir_valid = base_dir + 'images_set_validation_small/'
img_dir_test = base_dir + 'images_set_test/'
#img_dir_train_small = base_dir + 'images_set_very_small/'
#img_dir_valid = base_dir + 'images_set_validation_very_small/'
img_dir_train = img_dir_train_small
trained_models_dir = base_dir + 'trained_models/'

#img_target_size = 204
#img_target_size = 100
#img_target_size = 28
img_target_size = 996

# load training images
train_images = []
for filename in os.listdir(img_dir_train):
    if not filename.endswith('.png'):
        continue
    img_path = '{}{}'.format(img_dir_train, filename)
    img = image.img_to_array(image.load_img(img_path,
        target_size=(img_target_size, img_target_size)))
    #img = cv2.imread(img_path, 0)
    train_images.append(img)
x_train = np.asarray(train_images)

test_images = []
for filename in os.listdir(img_dir_test):
    if not filename.endswith('.png'):
        continue
    img_path = '{}{}'.format(img_dir_test, filename)
    img = image.img_to_array(image.load_img(img_path,
        target_size=(img_target_size, img_target_size)))
    test_images.append(img)
x_test = np.asarray(test_images)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

img_width, img_height = img_target_size, img_target_size
def AE_CNN():
    input_img = Input(shape=(img_width, img_height, 3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same',
            strides=2)(input_img)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    if img_target_size >= 100:
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
    #decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    #x = Conv2D(16, (3, 3), activation='relu', padding='same',
    #        strides=2)(input_img)
    #x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x)
    #encoded = Conv2D(32, (2, 2), activation='relu', padding="same",
    #        strides=2)(x)
    #x = Conv2D(32, (2, 2), activation='relu', padding="same")(encoded)
    #x = UpSampling2D((2, 2))(x)
    #x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #x = UpSampling2D((2, 2))(x)
    #x = Conv2D(16, (3, 3), activation='relu')(x)
    #x = UpSampling2D((2, 2))(x)
    #decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return Model(input_img, decoded)

trained_model_name = '{}model_weights_ae_cnn.h5'.format(trained_models_dir)
checkpoint_cnn = ModelCheckpoint(filepath = trained_model_name,
        save_best_only=True,monitor="val_loss", mode="min" )
history_cnn = History()
autoencoder_cnn = AE_CNN()
autoencoder_cnn.compile(optimizer='adam', loss='binary_crossentropy')
#autoencoder_cnn.compile(optimizer='adam', loss='mse')
autoencoder_cnn.fit(x_train, x_train, epochs=_epochs, batch_size=_batch_size,
        validation_split=0.1,
        verbose=0, callbacks=[history_cnn, checkpoint_cnn])

