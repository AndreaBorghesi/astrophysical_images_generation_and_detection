'''
Andrea Borghesi
University of Bologna
    2019-02-26

Given a train set of images (results of astrophysical simulations), attempt to
generate images with similar distribution
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
img_dir_train = base_dir + 'images_set/'
img_dir_train_small = base_dir + 'images_set_small/'
img_dir_valid = base_dir + 'images_set_validation_small/'
img_dir_test = base_dir + 'images_set_test/'
#img_dir_train_small = base_dir + 'images_set_very_small/'
#img_dir_valid = base_dir + 'images_set_validation_very_small/'
#img_dir_train = img_dir_train_small
trained_model_dir = base_dir + 'trained_models/'

# TODO: make experiments with data augmentation for training set
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

#img_target_size = 204
img_target_size = 100
img_target_size = 28
img_target_size = 996
#img_target_size = 244

enhanced_contrast = 0
#enhanced_contrast = -10

img_width, img_height = img_target_size, img_target_size

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

# load training images
#train_images = []
#idx = 0
#for i in tqdm(os.listdir(img_dir_train)):
#    if idx == 30:
#        break
#    img_path = os.path.join(img_dir_train, i)
#    if enhanced_contrast != 0:
#        img_enhanced = change_contrast_multi(Image.open(img_path),
#                [enhanced_contrast])
#        img = image.img_to_array(img_enhanced)
#    else:
#        img = image.img_to_array(image.load_img(img_path,
#            target_size=(img_target_size, img_target_size)))
#    train_images.append(img)
#    idx += 1
#x_train = np.asarray(train_images)
#
#print(x_train.nbytes)
#sys.exit()

#validation_images = []
#for i in tqdm(os.listdir(img_dir_valid)):
#    img_path = os.path.join(img_dir_valid, i)
#    img = image.img_to_array(image.load_img(img_path,
#        target_size=(img_target_size, img_target_size)))
#    validation_images.append(img)
#x_validation = np.asarray(validation_images)

test_images = []
for i in tqdm(os.listdir(img_dir_test)):
    img_path = os.path.join(img_dir_test, i)
    if enhanced_contrast != 0:
        img_enhanced = change_contrast_multi(Image.open(img_path),
                [enhanced_contrast])
        img = image.img_to_array(img_enhanced)
    else:
        img = image.img_to_array(image.load_img(img_path,
            target_size=(img_target_size, img_target_size)))
    test_images.append(img)
x_test = np.asarray(test_images)

#imgs = change_contrast_multi(Image.open(
#    img_dir_test + 'z_add_pow-3.0882_seed57212.png'), 
#        [-100, 0, -10, -25, -50])
#plt.imshow(imgs)
#plt.show()
#sys.exit()

#(x_train, _), (x_test, _) = mnist.load_data()
#x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#x_train = np.reshape(x_train, (len(x_train), img_target_size, img_target_size,
#    1))
#x_train.reshape((1000, 1000, 1))
#print(x_train.shape)
#sys.exit()

#x_train = np.reshape(x_train, (len(x_train), img_target_size, img_target_size, 1))
#x_test = np.reshape(x_test, (len(x_test), img_target_size, img_target_size, 1))
#print(x_train.shape)
#print(x_test.shape)
# train_datagen = ImageDataGenerator(rescale=1./255)

##
##outfile = 'imgs_as_np_array_file.npy'
##np.save(outfile, x)
#
##sys.exit()
#

#def AE_CNN():
#    input_img = Input(shape=(img_width, img_height, 3))
#    x = Conv2D(16, (3, 3), activation='relu', padding='same',
#            strides=2)(input_img)
#    x = MaxPooling2D((2,2), padding='same')(x)
#    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#    x = MaxPooling2D((2, 2), padding='same')(x)
#    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#    encoded = MaxPooling2D((2, 2), padding='same')(x)
#
#    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
#    x = UpSampling2D((2, 2))(x)
#    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#    x = UpSampling2D((2, 2))(x)
#    x = Conv2D(16, (3, 3), activation='relu')(x)
#    x = UpSampling2D((2, 2))(x)
#    if img_target_size >= 100:
#        x = Conv2D(16, (3, 3), activation='relu')(x)
#        x = UpSampling2D((2, 2))(x)
#    if img_target_size == 28:
#        x = Conv2D(16, (3, 3), activation='relu')(x)
#        x = UpSampling2D((3, 3))(x)
#        x = Conv2D(16, (3, 3), activation='relu')(x)
#        #x = UpSampling2D((2, 2))(x)
#
#    #decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
#    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
#
#    #x = Conv2D(16, (3, 3), activation='relu', padding='same',
#    #        strides=2)(input_img)
#    #x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x)
#    #encoded = Conv2D(32, (2, 2), activation='relu', padding="same",
#    #        strides=2)(x)
#    #x = Conv2D(32, (2, 2), activation='relu', padding="same")(encoded)
#    #x = UpSampling2D((2, 2))(x)
#    #x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
#    #x = UpSampling2D((2, 2))(x)
#    #x = Conv2D(16, (3, 3), activation='relu')(x)
#    #x = UpSampling2D((2, 2))(x)
#    #decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
#
#    return Model(input_img, decoded)
#
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
#autoencoder_cnn.fit(x_train, x_train, epochs=_epochs, batch_size=_batch_size,
##autoencoder_cnn.fit(x_train, x_train, epochs=5, batch_size=128,
#        validation_split=0.1,
#        verbose=1, callbacks=[history_cnn, checkpoint_cnn])

model_to_load = (trained_model_dir +
        'model_weights_ae_cnn_996imgSize_100ep_32bs_3nbch_0enhC.h5')
autoencoder_cnn = load_model(model_to_load)
decoded_imgs = autoencoder_cnn.predict(x_test)

n = 15
#plt.figure()
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    #reshaped_test = np.reshape(x_test[i], (img_target_size, img_target_size))
    #plt.imshow(reshaped_test)
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + n+1)
    #reshaped_dec_img = np.reshape(decoded_imgs[i], 
    #        (img_target_size, img_target_size))
    #plt.imshow(reshaped_dec_img)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


