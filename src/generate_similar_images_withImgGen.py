'''
Andrea Borghesi
University of Bologna
    2019-03-06

Given a train set of images (results of astrophysical simulations), attempt to
generate images with similar distribution
- use a image generator to load the train/test set
'''

#!/usr/bin/python

import numpy as np
import sys
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib import cm
import pickle
from pathlib import Path
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import math
import time
import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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
from tqdm import tqdm
from keras.datasets import mnist
import cv2
from PIL import Image

_batch_size = 32
_epochs = 100

base_dir = '/home/b0rgh/collaborations_potential/'
base_dir += 'mpasquato_AE_detection_astrophysics_images/'
img_dir_train = base_dir + 'img_generator/train/'
img_dir_validation = base_dir + 'img_generator/validation/'
img_dir_test = base_dir + 'img_generator/test/'
trained_models_dir = base_dir + 'trained_models/'
generated_imgs_dir = base_dir + 'generated_images/'

# TODO: make experiments with data augmentation for training set
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

#img_target_size = 204
#img_target_size = 100
img_target_size = 996
#img_target_size = 28

img_width, img_height = img_target_size, img_target_size
nb_channels = 3

enhanced_contrast = 0

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

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(img_dir_train,
        target_size=(img_width, img_height), batch_size=_batch_size,
        class_mode=None, shuffle=True)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(img_dir_validation,
        target_size=(img_width, img_height), batch_size=_batch_size,
        class_mode=None, shuffle=True)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(img_dir_test,
        target_size=(img_width, img_height), batch_size=_batch_size,
        class_mode=None, shuffle=True)

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
    if img_target_size == 28:
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((3, 3))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        #x = UpSampling2D((2, 2))(x)

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

#checkpoint_cnn = ModelCheckpoint(filepath = "model_weights_ae_cnn.h5",
#        save_best_only=True,monitor="val_loss", mode="min" )
#history_cnn = History()
#autoencoder_cnn = AE_CNN()
#autoencoder_cnn.summary()
#autoencoder_cnn.compile(optimizer='adam', loss='binary_crossentropy')
##autoencoder_cnn.compile(optimizer='adadelta', loss='mse')
##autoencoder_cnn.fit_generator(fixed_generator(train_generator_cnn),
##        samples_per_epoch=math.floor(41322 / _batch_size), nb_epoch=_epochs,
##        validation_data=fixed_generator(validation_generator_cnn),
##        nb_val_samples=math.floor(13877 / _batch_size),
##        verbose=1, callbacks=[history_cnn, checkpoint_cnn])
#
##autoencoder = Model(input_img, decoded)
##autoencoder_cnn.compile(optimizer='adam', loss='mse')
##autoencoder.fit(x, x, epochs=_epochs, batch_size=_batch_size, callbacks=None )
#
##autoencoder_cnn.fit(x_train, x_train, epochs=_epochs, batch_size=_batch_size,
###autoencoder_cnn.fit(x_train, x_train, epochs=5, batch_size=128,
##        validation_split=0.1,
##        verbose=1, callbacks=[history_cnn, checkpoint_cnn])
#
#autoencoder_cnn.fit_generator(fixed_generator(train_generator), 
#        steps_per_epoch=2000 // _batch_size,
#        epochs=_epochs, validation_data=fixed_generator(validation_generator),
#        validation_steps=800 // _batch_size,
#        verbose=1, callbacks=[history_cnn, checkpoint_cnn])

model_to_load = ('{}model_weights_ae_cnn_{}imgSize_{}ep_{}bs_{}nbch_{}enhC_'
        '{}_{}_{}.h5'.format(trained_models_dir, img_target_size, _epochs, 
            _batch_size, nb_channels, enhanced_contrast, train_loss, 
            imgGen_class_mode_str, n_gpu))
autoencoder_cnn = load_model(model_to_load)
print(model_to_load)
autoencoder_cnn.summary()

x_test = test_generator.next()
decoded_imgs = autoencoder_cnn.predict(x_test)

#n = 10
##plt.figure()
#plt.figure(figsize=(20, 4))
#for i in range(n):
#    ax = plt.subplot(2, n, i+1)
#    #reshaped_test = np.reshape(x_test[i], (img_target_size, img_target_size))
#    #plt.imshow(reshaped_test)
#    plt.imshow(x_test[i])
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#
#    ax = plt.subplot(2, n, i + n+1)
#    #reshaped_dec_img = np.reshape(decoded_imgs[i], 
#    #        (img_target_size, img_target_size))
#    #plt.imshow(reshaped_dec_img)
#    plt.imshow(decoded_imgs[i])
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()

n = 45
for i in range(n):
    print(i)
    fig = plt.figure()
    plt.imshow(x_test[i])
    fig_test = ('{}test_ae_cnn_{}imgSize_{}ep_{}bs_{}nbch_{}enhC_'
        '{}_{}_{}_{}.png'.format(generated_imgs_dir, img_target_size, _epochs, 
            _batch_size, nb_channels, enhanced_contrast, train_loss, 
            imgGen_class_mode_str, n_gpu, i))
    plt.savefig(fig_test)
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(decoded_imgs[i])
    fig_gen = ('{}gen_ae_cnn_{}imgSize_{}ep_{}bs_{}nbch_{}enhC_'
        '{}_{}_{}_{}.png'.format(generated_imgs_dir, img_target_size, _epochs, 
            _batch_size, nb_channels, enhanced_contrast, train_loss, 
            imgGen_class_mode_str, n_gpu, i))
    plt.savefig(fig_gen)
    plt.close(fig)


