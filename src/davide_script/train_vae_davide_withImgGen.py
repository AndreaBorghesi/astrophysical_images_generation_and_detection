'''
Andrea Borghesi
University of Bologna
    2019-03-26

Script to train the VAE using DAVIDE 
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
from keras.layers import UpSampling1D, Lambda, Dropout, merge, Reshape
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, History
from PIL import Image

base_dir = '/davide/home/userexternal/aborghes/'
base_dir += 'astrophysical_images_generation_and_detection/'
trained_models_dir = base_dir + 'trained_models/'
img_dir_train = base_dir + 'img_generator/train/'
img_dir_validation = base_dir + 'img_generator/validation/'
img_dir_test = base_dir + 'img_generator/test/'

img_target_size = 996
#img_target_size = 244
#img_target_size = 512

img_width, img_height = img_target_size, img_target_size
nb_channels = 3

enhanced_contrast = 0
#enhanced_contrast = -10

_batch_size = 16
_epochs = 100
_latent_dim = 2
_cnn = True
_vae_loss_kl_weight = 1
_vae_loss_recon_weight = 1
_bn = True

imgGen_class_mode = 'input'
#imgGen_class_mode = None
if imgGen_class_mode == None:
    imgGen_class_mode_str = 'None'
else:
    imgGen_class_mode_str = 'input'

#train_loss = 'binary_crossentropy'
#train_loss = 'mae'
train_loss = 'mean_squared_error'

n_gpu = 4

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

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(img_dir_train,
        target_size=(img_width, img_height), batch_size=_batch_size,
        class_mode=imgGen_class_mode, shuffle=True)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(img_dir_validation,
        target_size=(img_width, img_height), batch_size=_batch_size,
        class_mode=imgGen_class_mode, shuffle=True)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(img_dir_test,
        target_size=(img_width, img_height), batch_size=_batch_size,
        class_mode=imgGen_class_mode, shuffle=True)

original_dim = img_target_size * img_target_size
if not _cnn:
    input_shape = (original_dim, )
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
    if _bn:
        x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    if _bn:
        x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    shape = K.int_shape(x)
    x = Flatten()(x)

    mu = Dense(_latent_dim, name='z_mean')(x)
    log_sigma = Dense(_latent_dim, name='z_log_var')(x)

    z = Lambda(sampling, output_shape=(_latent_dim,))([mu, log_sigma])
    encoder = Model(inputs, [mu, log_sigma, z], name='encoder_cnn')

    latent_inputs = Input(shape=(_latent_dim,), name='z_sampling')

    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    if _bn:
        x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    if _bn:
        x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    if _bn:
        x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    if _bn:
        x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    #if img_target_size >= 100:
    #    x = Conv2D(16, (3, 3), activation='relu')(x)
    #    x = UpSampling2D((2, 2))(x)
    if _bn:
        x = BatchNormalization()(x)
    outputs = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    decoder = Model(latent_inputs, outputs, name='decoder_cnn')
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_cnn')

    #encoder.summary()
    #decoder.summary()
    #vae.summary()

def vae_loss(y_true, y_pred):
    recon = K.mean(K.square(y_pred - y_true))
    kl = 0.5*K.sum(
            K.exp(2*log_sigma) + K.square(mu) -1-2 * log_sigma,axis=1)
    return (_vae_loss_recon_weight * original_dim * recon + 
            _vae_loss_kl_weight*kl)

if _cnn:
    cnn_str = '_cnn'
else:
    cnn_str = ''
if _bn:
    bn_str = '_bn'
else:
    bn_str = ''

model_weights = ('{}model_weights_vae{}{}_{}imgSize_{}ep_{}bs_{}nbch_{}enhC_'
        '{}_{}_{}_{}zdim_{}recW_{}klW.h5'.format(trained_models_dir, cnn_str, 
            bn_str, img_target_size, _epochs, _batch_size, nb_channels, 
            enhanced_contrast, train_loss, imgGen_class_mode_str, n_gpu, 
            _latent_dim, _vae_loss_recon_weight, _vae_loss_kl_weight))

checkpoint = ModelCheckpoint(
        filepath = "model_weights_vae{}.h5".format(cnn_str),
        save_best_only=True,monitor="val_loss", mode="min" )
history = History()

vae.compile(optimizer='adam', loss=vae_loss)

print("VAE{} Created & Compiled".format(cnn_str))

before_training_time = time.time()
if imgGen_class_mode_str == 'input':
    vae.fit_generator(train_generator, 
            steps_per_epoch=2000 // _batch_size,
            epochs=_epochs, validation_data=validation_generator,
            validation_steps=800 // _batch_size,
            verbose=1, callbacks=[history, checkpoint])
else:
    vae.fit_generator(fixed_generator(train_generator), 
            steps_per_epoch=2000 // _batch_size,
            epochs=_epochs, validation_data=fixed_generator(validation_generator),
            validation_steps=800 // _batch_size,
            verbose=1, callbacks=[history, checkpoint])

after_training_time = time.time()
train_time = after_training_time - before_training_time
print("VAE{} Trained (in {} s)".format(cnn_str, train_time))

model_saved = ('{}model_vae{}{}_{}imgSize_{}ep_{}bs_{}nbch_{}enhC_{}_{}_{}'
        '_{}zdim_{}recW_{}klW.h5'.format(trained_models_dir, cnn_str, bn_tr, 
            img_target_size, _epochs, _batch_size, nb_channels, 
            enhanced_contrast, train_loss, imgGen_class_mode_str, n_gpu, 
            _latent_dim, _vae_loss_recon_weight, _vae_loss_kl_weight))
vae.save(model_saved)
