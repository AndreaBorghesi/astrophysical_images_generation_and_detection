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
from keras.preprocessing import image
from tqdm import tqdm

_batch_size = 32
_epochs = 100

img_dir = '/home/b0rgh/collaborations_potential/'
#img_dir += 'mpasquato_AE_detection_astrophysics_images/images_set/'
img_dir += 'mpasquato_AE_detection_astrophysics_images/images_set_small/'

# TODO: make experiments with data augmentation for training set
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# load training images
train_images = []
for i in tqdm(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, i)
    img = image.img_to_array(image.load_img(img_path, target_size=(512, 512)))
    train_images.append(img)
x = np.asarray(train_images)






