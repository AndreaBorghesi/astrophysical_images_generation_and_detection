'''
Andrea Borghesi
University of Bologna
    2019-02-25

Attempt to discern 'good' astrophysical simulations from 'bad' ones using
autoencoders
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

_batch_size = 32
_epochs = 100

df = pd.read_csv('../df', sep=' ')

img_names = df['X1']

del df['X1']

scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)

n_samples, n_features = df.shape

msk = np.random.rand(n_samples) < 0.7
x_train = df[msk]
x_test = df[~msk]

input_data = Input(shape=(n_features,))

encoded = Dense(n_features * 10, activation='relu',
        activity_regularizer=regularizers.l1(1e-5))(input_data)
decoded = Dense(n_features, activation='linear')(encoded)

autoencoder = Model(input_data, decoded)

autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
history = autoencoder.fit(x_train, x_train, epochs=_epochs, 
        batch_size=_batch_size, shuffle=True, 
        validation_split=0.1,
        verbose=1)







