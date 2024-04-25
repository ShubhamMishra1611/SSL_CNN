from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import IPython.display as ipd
from pathlib import Path
from tqdm import tqdm
from sklearn.utils import shuffle
import random
import sklearn
import re 


import tensorflow as tf

from tensorflow import keras

filters = 64
kernel_size = (3,3)
strides = (1,1)
input_shape = (14, 511, 10)
rate = 0.5
K = 36


def get_model():
    model = keras.Sequential ([
    # input layer (14 * 511 * 10) (convolutional layers + batch normalization (BN) w ReLU)
    keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, input_shape=input_shape, activation='relu', name='conv1'),
    keras.layers.BatchNormalization(name = 'bn1'),

    # 2nd convolutional layers + batch normalization (BN) w ReLU
    keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation='relu', name='conv2'),
    keras.layers.BatchNormalization(name = 'bn2'),
    # dropout procedure with rate 0.5
    tf.keras.layers.Dropout(rate, name = 'dn1'),

    # 1st fully connected layer w ReLU & dropout procedure with rate 0.5
    keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(512, activation = 'relu', name = 'fc1'),
    tf.keras.layers.Dropout(rate, name = 'dn2'),

    # 2nd fully connected layer w ReLU & dropout procedure with rate 0.5
    tf.keras.layers.Dense(512, activation = 'relu', name = 'fc2'),
    tf.keras.layers.Dropout(rate, name = 'dn3'),

    # Output layer
    tf.keras.layers.Dense(K, activation = 'softmax', name = 'output'),
    ])

    model.summary()

    return model