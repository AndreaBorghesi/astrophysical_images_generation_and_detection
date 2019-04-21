'''
Andrea Borghesi
University of Bologna
    2019-03-09

Given a train set of images (results of astrophysical simulations), attempt to
generate images with similar distribution
- use a image generator to load the train/test set
- VAE 
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
from keras.layers import UpSampling1D, Lambda, Dropout, merge, Reshape
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
_epochs = 2
_latent_dim = 3

base_dir = '/home/b0rgh/collaborations_potential/'
base_dir += 'mpasquato_AE_detection_astrophysics_images/'
img_dir_train = base_dir + 'img_generator_small/train/'
img_dir_validation = base_dir + 'img_generator_small/validation/'
img_dir_test = base_dir + 'img_generator_small/test/'
trained_model_dir = base_dir + 'trained_models/'

img_target_size = 100
img_target_size = 996
#img_target_size = 28

img_width, img_height = img_target_size, img_target_size
nb_channels = 3

enhanced_contrast = 0

_cnn = True

_vae_loss_kl_weight = 1
_vae_loss_recon_weight = 1

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

def sampling(args):
   z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

if _cnn:
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(img_dir_train,
            target_size=(img_width, img_height), batch_size=_batch_size,
            class_mode=None, shuffle=True)
    validation_datagen = train_datagen
    validation_generator = validation_datagen.flow_from_directory(
            img_dir_validation,
            target_size=(img_width, img_height), batch_size=_batch_size,
            class_mode=None, shuffle=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(img_dir_test,
            target_size=(img_width, img_height), batch_size=_batch_size,
            class_mode=None, shuffle=True)
else:
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(img_dir_train,
            target_size=(img_width, img_height), batch_size=_batch_size,
            class_mode=None, shuffle=True)
    validation_datagen = train_datagen
    validation_generator = validation_datagen.flow_from_directory(
            img_dir_validation,
            target_size=(img_width, img_height), batch_size=_batch_size,
            class_mode=None, shuffle=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(img_dir_test,
            target_size=(img_width, img_height), batch_size=_batch_size,
            class_mode=None, shuffle=True)

original_dim = img_target_size * img_target_size
if not _cnn:
    #input_img = Input(shape=(img_width, img_height, 3))
    input_shape = (original_dim, )
    #input_shape = (img_width, img_height, 3)

    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    mu = Dense(_latent_dim, name='z_mean')(x)
    log_sigma = Dense(_latent_dim, name='z_log_var')(x)

    z = Lambda(sampling, output_shape=(_latent_dim,))([mu, log_sigma])

    encoder = Model(inputs, [mu, log_sigma, z], name='encoder')

    latent_inputs = Input(shape=(_latent_dim,), name='z_sampling')

    x = Dense(16, activation='relu')(latent_inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

else:
    input_shape = (img_width, img_height, 3)
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Conv2D(16, (3, 3), activation='relu', padding='same',
            strides=2)(inputs)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    shape = K.int_shape(x)
    x = Flatten()(x)

    mu = Dense(_latent_dim, name='z_mean')(x)
    log_sigma = Dense(_latent_dim, name='z_log_var')(x)

    z = Lambda(sampling, output_shape=(_latent_dim,))([mu, log_sigma])
    encoder = Model(inputs, [mu, log_sigma, z], name='encoder_cnn')

    encoder.summary()

    latent_inputs = Input(shape=(_latent_dim,), name='z_sampling')

    print(shape)

    print(latent_inputs)

    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    print(x)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    print(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    if img_target_size >= 100:
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
    #if img_target_size == 28:
    #    x = Conv2D(16, (3, 3), activation='relu')(x)
    #    x = UpSampling2D((3, 3))(x)
    #    x = Conv2D(16, (3, 3), activation='relu')(x)
    outputs = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

    decoder = Model(latent_inputs, outputs, name='decoder_cnn')
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_cnn')

    decoder.summary()
    vae.summary()


def vae_loss(y_true, y_pred):
    recon = K.mean(K.square(y_pred - y_true))
    kl = 0.5*K.sum(
            K.exp(2*log_sigma) + K.square(mu) -1-2 * log_sigma,axis=1)
    return (_vae_loss_recon_weight * original_dim * recon + 
            _vae_loss_kl_weight*kl)

if _cnn:
    cnn_str = '_CNN_'
else:
    cnn_str = ''
checkpoint_vae = ModelCheckpoint(
        filepath = "model_weights_vae{}.h5".format(cnn_str),
        save_best_only=True,monitor="val_loss", mode="min" )
history_vae = History()

#vae.compile(optimizer='adam', loss=vae_loss)
#vae.fit_generator(fixed_generator(train_generator), 
#        steps_per_epoch=2000 // _batch_size,
#        epochs=_epochs, validation_data=fixed_generator(validation_generator),
#        validation_steps=800 // _batch_size,
#        verbose=1, callbacks=[history_vae, checkpoint_vae])

model_to_load = (trained_model_dir + 'model_weights_vae_cnn_bn_996imgSize_'
    '100ep_32bs_3nbch_0enhC_input_4_50zdim_1recW_1klW.hdf5')

vae_cnn = load_model(model_to_load, custom_objects={'vae_loss': vae_loss})

x_test = test_generator.next()
decoded_imgs = vae_cnn.predict(x_test)

n = 10
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


