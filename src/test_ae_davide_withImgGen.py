'''
Andrea Borghesi
University of Bologna
    2019-03-11

Script to test the trained N
- use a image generator to load the train/test set
'''

#!/usr/bin/python

import numpy as np
import sys
import os
import subprocess
import math
import time
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
from PIL import Image

_batch_size = 32
_epochs = 100

base_dir = '/davide/home/userexternal/aborghes/'
base_dir += 'astrophysical_images_generation_and_detection/'
trained_models_dir = base_dir + 'trained_models/'
img_dir_train = base_dir + 'img_generator/train/'
img_dir_validation = base_dir + 'img_generator/validation/'
img_dir_test = base_dir + 'img_generator/test/'

#img_target_size = 204
#img_target_size = 100
#img_target_size = 28
img_target_size = 996
#img_target_size = 244
img_width, img_height = img_target_size, img_target_size

img_width, img_height = img_target_size, img_target_size
nb_channels = 3

enhanced_contrast = 0
#enhanced_contrast = -10

imgGen_class_mode = 'input'
#imgGen_class_mode = None
if imgGen_class_mode == None:
    imgGen_class_mode_str = 'None'
else:
    imgGen_class_mode_str = 'input'

#train_loss = 'binary_crossentropy'
train_loss = 'mae'
#train_loss = 'mean_squared_error'

n_gpu = 1

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        value = 128 + factor * (c - 128)
        return max(0, min(255, value))
    return img.point(contrast)

def change_contrast_multi(img, steps):
    width, height = img.size
    canvas = Image.new('RGB', (width * len(steps), height))
    for n, level in enumerate(steps):
        img_filtered = change_contrast(img, level)
        canvas.paste(img_filtered, (width * n, 0))
    return canvas

def flattened_generator(generator):
    for batch in generator:
        yield (batch.reshape(-1,img_width*img_height*nb_channels),
                batch.reshape(-1,img_width*img_height*nb_channels))

def fixed_generator(generator):
    for batch in generator:
        yield (batch, batch)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(img_dir_test,
        target_size=(img_width, img_height), batch_size=_batch_size,
        class_mode=imgGen_class_mode, shuffle=True)

model_to_load = ('{}model_weights_ae_cnn_{}imgSize_{}ep_{}bs_{}nbch_{}enhC_'
        '{}_{}_{}.h5'.format(trained_models_dir, img_target_size, _epochs, 
            _batch_size, nb_channels, enhanced_contrast, train_loss, 
            imgGen_class_mode_str, n_gpu))
autoencoder_cnn = load_model(model_to_load)
print(model_to_load)
autoencoder_cnn.summary()

x_test = fixed_generator(test_generator.next())
decoded_imgs = autoencoder_cnn.predict(x_test)

