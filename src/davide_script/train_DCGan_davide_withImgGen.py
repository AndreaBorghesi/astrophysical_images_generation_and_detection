'''
Andrea Borghesi
University of Bologna
    2019-06-13

Given a train set of images (results of astrophysical simulations), attempt to
generate images with similar distribution
- use a image generator to load the train/test set
- DCGan
** DAVIDE script **
- inspired by: https://github.com/mrdragonbear/GAN-Tutorial
'''

#!/usr/bin/python

import numpy as np
import sys
import os
import subprocess
import math
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import tensorflow as tf
from keras.models import load_model, Sequential, Model
from keras import backend as K 
from keras.optimizers import Adam
from keras import optimizers, initializers, regularizers
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Input
from keras.layers import Cropping2D, LeakyReLU, AveragePooling2D
from keras.layers import UpSampling1D, Lambda, Dropout, merge, Reshape
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Conv2D, UpSampling2D, Activation, Conv2DTranspose
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, History
from PIL import Image
from keras.initializers import RandomNormal
from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from matplotlib import cm

_batch_size = 128
_epochs = 5000
_latent_dim = 100

base_dir = '/davide/home/userexternal/aborghes/'
base_dir += 'astrophysical_images_generation_and_detection/'
trained_models_dir = base_dir + 'trained_models/GAN/'
img_dir_train = base_dir + 'new_img_generator/train/'
img_dir_validation = base_dir + 'new_img_generator/validation/'
img_dir_test = base_dir + 'new_img_generator/test/'
aae_img_dir = base_dir + 'aae_generated_imgs/'

img_target_size = 996
#img_target_size = 64

img_width, img_height = img_target_size, img_target_size
nb_channels = 3

n_gpu = 4

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
validation_datagen = train_datagen
validation_generator = validation_datagen.flow_from_directory(
        img_dir_validation,
        target_size=(img_width, img_height), batch_size=_batch_size,
        class_mode=None, shuffle=True)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(img_dir_test,
        target_size=(img_width, img_height), batch_size=_batch_size,
        class_mode=None, shuffle=True)

def imagegrid_fullRes(epochnumber, images):
    for index,img in enumerate(images):
        #print(img_target_size)
        fig = plt.figure(figsize=(img_target_size/1000, img_target_size/1000),
                dpi=100)
        img = img.reshape((img_target_size, img_target_size, nb_channels))
        #img -= img.min()
        #img /= img.max()
        ax = fig.add_subplot(1,1,1)
        ax.set_axis_off()
        ax.imshow(img, cmap="gray",vmin=0,vmax=255)
        fig.savefig("{}DCGAN_{}_{}.png".format(aae_img_dir, epochnumber, 
            #index))
            index),dpi=1000)
        plt.close(fig)
    for index,img in enumerate(images):
        fig = plt.figure()
        #img = img.reshape((img_target_size, img_target_size, nb_channels))
        img = np.uint8((img+1)/2*255)
        #img -= img.min()
        #img /= img.max()
        ax = fig.add_subplot(1,1,1)
        ax.set_axis_off()
        ax.imshow(img, cmap="gray")
        fig.savefig("{}2_DCGAN_{}_{}.png".format(aae_img_dir,epochnumber,index))
        plt.close(fig)

def imagegrid(epochnumber, images):
    for index,img in enumerate(images):
        fig = plt.figure()
        img = img.reshape((img_target_size, img_target_size, nb_channels))
        #img = np.uint8((img+1)/2*255)
        img = np.uint8(img*255)
        #img -= img.min()
        #img /= img.max()
        ax = fig.add_subplot(1,1,1)
        ax.set_axis_off()
        #ax.imshow(img, cmap="gray",vmin=0,vmax=255)
        ax.imshow(img)
        fig.savefig("{}lowRes_DCGAN_{}_{}.png".format(aae_img_dir,epochnumber,index))
        plt.close(fig)

def generator(latent_dim=_latent_dim, leaky_alpha=0.2, init_stddev=0.02):
    g = Sequential()
    g.add(Dense(4*4*1024, input_dim=latent_dim, 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(Reshape((4, 4, 1024)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(512, kernel_size=5, strides=2, padding='same',
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(32, kernel_size=5, strides=1, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(3, kernel_size=4, strides=1, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(Activation('tanh'))
    print("--------- GENERATOR ------------")
    g.summary()
    return g

def discriminator(leaky_alpha=0.2, init_stddev=0.02):
    d = Sequential()
    d.add(Conv2D(16, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev),
        input_shape=(img_width, img_height, nb_channels)))
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Conv2D(32, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(BatchNormalization())
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Conv2D(64, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(BatchNormalization())
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Conv2D(128, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(BatchNormalization())
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Conv2D(256, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(BatchNormalization())
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Conv2D(512, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(BatchNormalization())
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Flatten())
    d.add(Dense(1, kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(Activation('sigmoid'))
    print("--------- DISCRIMINATOR ------------")
    d.summary()
    return d

def generator_fullRes(latent_dim=_latent_dim, leaky_alpha=0.2, init_stddev=0.02):
    g = Sequential()
    g.add(Dense(4*4*256, input_dim=latent_dim, 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(Reshape((4, 4, 256)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same',
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(32, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(16, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(8, kernel_size=5, strides=1, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(BatchNormalization())
    g.add(LeakyReLU(alpha=leaky_alpha))
    g.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    g.add(Cropping2D(cropping=14))
    g.add(Activation('tanh'))
    print("--------- GENERATOR ------------")
    g.summary()
    return g

def discriminator_fullRes(leaky_alpha=0.2, init_stddev=0.02):
    d = Sequential()
    d.add(Conv2D(16, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev),
        input_shape=(img_width, img_height, nb_channels)))
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Conv2D(32, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(BatchNormalization())
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Conv2D(64, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(BatchNormalization())
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Conv2D(64, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(BatchNormalization())
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Conv2D(128, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(BatchNormalization())
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Conv2D(128, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(BatchNormalization())
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Conv2D(256, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(BatchNormalization())
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Conv2D(512, kernel_size=5, strides=2, padding='same', 
        kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(BatchNormalization())
    d.add(LeakyReLU(alpha=leaky_alpha))
    d.add(Flatten())
    d.add(Dense(1, kernel_initializer=RandomNormal(stddev=init_stddev)))
    d.add(Activation('sigmoid'))
    print("--------- DISCRIMINATOR ------------")
    d.summary()
    return d

def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])  

def DCGAN(latent_dim=_latent_dim):
    g = generator(latent_dim)
    d = discriminator(0.2, 0.02)
    d.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')
    d.trainable = False
    gan = Sequential([g, d])
    gan.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), 
        loss='binary_crossentropy')
    return gan, g, d

def train(x_train, x_test, batch_size=_batch_size, epochs=_epochs, 
        eval_size=20, smooth=.1):
    print("--- Train ---")
    gan, g, d = DCGAN()

    y_train_real, y_train_fake = make_labels(batch_size)
    y_eval_real,  y_eval_fake  = make_labels(eval_size)

    half_batch = int(batch_size / 2)
    for e in range(epochs):
        print("Epoch {}".format(e))
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        noise = np.random.normal(0, 1, size=(batch_size, _latent_dim))
        generated_images = g.predict_on_batch(noise)

        # Train discriminator on generated images
        d.trainable = True
        d.train_on_batch(imgs, y_train_real*(1-smooth))
        d.train_on_batch(generated_images, y_train_fake)

        # Train generator
        d.trainable = False
        g_loss=gan.train_on_batch(noise, y_train_real)

        # evaluate
        if eval_size > len(x_test):
            eval_size = len(x_test.shape)
        idx = np.random.randint(0, x_test.shape[0], eval_size)
        eval_imgs = x_test[idx]
        x_eval_real = eval_imgs
        noise = np.random.normal(loc=0, scale=1, size=(eval_size, _latent_dim))
        x_eval_fake = g.predict_on_batch(noise)

        d_loss  = d.test_on_batch(x_eval_real, y_eval_real)
        d_loss += d.test_on_batch(x_eval_fake, y_eval_fake)
        g_loss  = gan.test_on_batch(noise, y_eval_real)
        print("Epoch: {:>3}/{} DLoss: {:>6.4f} GLoss: {:>6.4f}".format(
                        e+1, epochs, d_loss, g_loss))  

        # save weigths
        print("-- Saving models --")
        discr_mdl = ('{}mdl_dcgan_discr_{}imgSize_{}ep_{}bs'
                '_{}nbch_{}_{}zdim.hdf5'.format(
            trained_models_dir, img_target_size, _epochs,
            _batch_size, nb_channels, n_gpu, _latent_dim))
        gen_mdl = ('{}mdl_dcgan_gen_{}imgSize_{}ep_{}bs'
                '_{}nbch_{}_{}zdim.hdf5'.format(
            trained_models_dir, img_target_size, _epochs,
            _batch_size, nb_channels, n_gpu, _latent_dim))
        d.save(discr_mdl)
        g.save(gen_mdl)

        # plot generated images
        if e % 200 == 0:
            imagegrid(e, x_eval_fake)


print("Going to create & train DCGAN")
x_train = train_generator.next()
x_test = test_generator.next()
before_train = time.time()
train(x_train, x_test)
after_train = time.time()
print("DCGAN fully trained in {0:.3f}".format(after_train - before_train))

