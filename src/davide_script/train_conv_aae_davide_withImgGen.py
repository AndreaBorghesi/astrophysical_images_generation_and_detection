'''
Andrea Borghesi
University of Bologna
    2019-04-21

Given a train set of images (results of astrophysical simulations), attempt to
generate images with similar distribution
- use a image generator to load the train/test set
- AE within generative adversarial approach
- with convolution
'''

#!/usr/bin/python

import numpy as np
import sys
import os
import subprocess
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import tensorflow as tf
from keras.models import load_model, Sequential, Model
from keras import backend as K 
from keras.optimizers import Adam
from keras import optimizers, initializers, regularizers
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Input
from keras.layers import UpSampling1D, Lambda, Dropout, merge, Reshape
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, History
from PIL import Image
from keras.initializers import RandomNormal
from sklearn.neighbors.kde import KernelDensity

_batch_size = 32
_epochs = 100
_latent_dim = 20

base_dir = '/davide/home/userexternal/aborghes/'
base_dir += 'astrophysical_images_generation_and_detection/'
trained_models_dir = base_dir + 'trained_models/'
img_dir_train = base_dir + 'img_generator/train/'
img_dir_validation = base_dir + 'img_generator/validation/'
img_dir_test = base_dir + 'img_generator/test/'
aae_img_dir = base_dir + 'aae_generated_imgs/'

img_target_size = 996

img_width, img_height = img_target_size, img_target_size
nb_channels = 3

enhanced_contrast = 0

initializer = RandomNormal(mean=0.0, stddev=0.01, seed=None)

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

class GAE():
    def __init__(self, img_shape=(img_target_size,img_target_size,nb_channels), 
            encoded_dim=_latent_dim):
        self.encoded_dim = encoded_dim
        self.optimizer_reconst = Adam(0.01)
        self.optimizer_discriminator = Adam(0.01)
        self._initAndCompileFullModel(img_shape, encoded_dim)
        self.img_shape = img_shape

    def _genEncoderModel(self, img_shape, encoded_dim):
        encoder = Sequential()
        encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same',
            strides=2, input_shape=img_shape))
        encoder.add(MaxPooling2D((2,2), padding='same'))
        encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        encoder.add(MaxPooling2D((2, 2), padding='same'))
        encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        encoder.add(MaxPooling2D((2, 2), padding='same'))
        encoder.add(Flatten())
        encoder.add(Dense(encoded_dim))
        #encoder.add(BatchNormalization())
        #encoder.add(Activation('tanh'))
        encoder.summary()
        return encoder

    def _getDecoderModel(self, encoded_dim, img_shape):
        decoder = Sequential()
        decoder.add(Dense(8*3*3, input_dim=encoded_dim))
        #decoder.add(BatchNormalization())
        #decoder.add(Activation('tanh'))
        decoder.add(Reshape((3, 3, 8), input_shape=(8*3*3,)))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2D(16, (3, 3), activation='relu'))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2D(16, (3, 3), activation='relu'))
        decoder.add(UpSampling2D((2, 2)))
        #decoder.add(Activation('tanh'))
        decoder.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
        #decoder.add(Activation('tanh'))
        decoder.summary()
        return decoder

    #def _getDiscriminator(self, img_shape):
    #    discriminator = Sequential()
    #    discriminator.add(Conv2D(16, (3, 3), activation='relu', padding='same',
    #        strides=2, input_shape=img_shape))
    #    discriminator.add(MaxPooling2D((2,2), padding='same'))
    #    discriminator.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    #    discriminator.add(MaxPooling2D((2, 2), padding='same'))
    #    discriminator.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    #    discriminator.add(MaxPooling2D((2, 2), padding='same'))
    #    discriminator.add(Flatten())
    #    discriminator.add(Dense(1, activation='sigmoid',
    #        kernel_initializer=initializer, bias_initializer=initializer))
    #    return discriminator

    def _getDiscriminator(self, encoded_dim):
        discriminator = Sequential()
        discriminator.add(Dense(1000, activation='relu', input_dim=encoded_dim,
            kernel_initializer=initializer, bias_initializer=initializer))
        discriminator.add(Dense(1000, activation='relu',
            kernel_initializer=initializer, bias_initializer=initializer))
        discriminator.add(Dense(1, activation='sigmoid',
            kernel_initializer=initializer, bias_initializer=initializer))
        discriminator.summary()
        return discriminator

    def _initAndCompileFullModel(self, img_shape, encoded_dim):
        self.encoder = self._genEncoderModel(img_shape, encoded_dim)
        self.decoder = self._getDecoderModel(encoded_dim, img_shape)
        #self.discriminator = self._getDiscriminator(img_shape)
        self.discriminator = self._getDiscriminator(encoded_dim)
        img = Input(shape=img_shape)
        encoded_repr = self.encoder(img)
        gen_img = self.decoder(encoded_repr)
        self.autoencoder = Model(img, gen_img)
        valid = self.discriminator(encoded_repr)
        self.encoder_discriminator = Model(img, valid)
        self.discriminator.compile(optimizer=self.optimizer_discriminator,
                loss='binary_crossentropy', metrics=['accuracy'])
        self.autoencoder.compile(optimizer=self.optimizer_reconst, loss ='mse')
        for layer in self.discriminator.layers:
            layer.trainable = False
        #self.discriminator.compile(optimizer=self.optimizer_discriminator,
        #        loss='binary_crossentropy', metrics=['accuracy'])
        self.encoder_discriminator.compile(optimizer=self.optimizer_discriminator,
                loss='binary_crossentropy', metrics=['accuracy'])

        print("---- ENCODER ----")
        self.encoder.summary()
        print("---- DECCODER ----")
        self.decoder.summary()
        print("---- DISCR ----")
        self.discriminator.summary()
        print("---- ENCODER_DISCR ----")
        self.encoder_discriminator.summary()
        print("---- AUTOENCODER ----")
        self.autoencoder.summary()

    def imagegrid(self, epochnumber):
        images = self.generateImages(10)
        for index,img in enumerate(images):
            fig = plt.figure(figsize=(img_target_size, img_target_size))
            img = img.reshape((img_target_size, img_target_size, nb_channels))
            ax = fig.add_subplot(1,1,1)
            ax.set_axis_off()
            ax.imshow(img, cmap="gray")
            fig.savefig("{}AAE_{}_{}.png".format(aae_img_dir, epochnumber, 
                index))
            plt.close(fig)

    def generateImages(self, n=10):
         latents = 5*np.random.normal(size=(n, self.encoded_dim))
         imgs = self.decoder.predict(latents)
         return imgs

    def train(self, x_train, batch_size=_batch_size, epochs=_epochs):
        half_batch = int(batch_size / 2)
        checkpoint_gan_ae = ModelCheckpoint(
            filepath = "model_weights_gan_ae.h5", mode="auto")

        for epoch in range(epochs):
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            imgs = x_train[idx]
            latent_fake = self.encoder.predict(imgs)
            latent_real = 5*np.random.normal(size=(half_batch,
                self.encoded_dim))
            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            valid_y = np.ones((batch_size, 1))
            g_loss_reconstruction = self.autoencoder.train_on_batch(imgs, imgs)
            g_logg_similarity = self.encoder_discriminator.train_on_batch(imgs,
                     valid_y)
            print ("%d [D loss: %f, acc: %.2f%%] [G acc: %f, mse: %f]" % (epoch,
                d_loss[0], 100*d_loss[1], g_logg_similarity[1], 
                g_loss_reconstruction))

            if epoch % 10 == 0:
                self.imagegrid(epoch)

        #self.autoencoder.fit_generator(fixed_generator(train_generator), 
        #    steps_per_epoch=2000 // _batch_size,
        #    epochs=_epochs, validation_data=fixed_generator(validation_generator),
        #    validation_steps=800 // _batch_size,
        #    verbose=1, callbacks=[checkpoint_gan_ae])


ann = GAE(img_shape=(img_target_size,img_target_size, nb_channels), 
        encoded_dim=_latent_dim)
x_train = train_generator.next()
ann.train(x_train, epochs=_epochs)


