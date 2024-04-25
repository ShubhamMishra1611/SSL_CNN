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
plt.style.use('seaborn')


# try:
#   # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x
# except Exception:
#   pass
import tensorflow as tf

from tensorflow import keras

DEBUG = False

print(tf.__version__)

from extract_audio import extractAudiodata
from utils import sorted_nicely
from calcs import featureExtractor2, computeSI
from model import get_model
import logging
import os
logging.basicConfig(filename='ssl_log_visualize.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

dir_path = 'D:\sem wise course\sem6\intro to acoustics\sem project\CNNbased\CNN_DOA\DATASETS\MATLAB_TEST10'
audiofiles = [str(file) for file in Path(dir_path).glob('SA*.wav')]
print(len(audiofiles))
audiofiles = sorted_nicely(audiofiles)

logging.info(f'Fetching some random files {random.choices(audiofiles, k=10)}')
print(f'Fetching some random files {random.choices(audiofiles, k=10)}')
sr = 16000
duration = 0.5

audioMatrix, sourceIDs, labels30, labels10 = extractAudiodata(audiofiles, sr*duration)

logging.info(f'audioMatrix has shape {audioMatrix.shape} where {audioMatrix.shape[0]} is length in samples, {audioMatrix.shape[1]} are the microphones and {audioMatrix.shape[2]} are the number of setences')

print('audioMatrix has shape {} where {} is length in samples, {} are the microphones and {} are the number of setences'.format(audioMatrix.shape, audioMatrix.shape[0], audioMatrix.shape[1], audioMatrix.shape[2]))

logging.info(f'sourceIDs is an array of length {sourceIDs.shape[0]}, one for each sentence')
print('labels10 is an array of length {}, one for each sentence\n'.format(labels10.shape[0]))

# Visualizing some examples: only from the first microphone
plt.figure(figsize=(15, 10))
time_axis = np.arange(0,sr*duration) / sr
for i in range(0, 10):
  plt.subplot(5, 2, i+1)
  plt.plot(time_axis, audioMatrix[:,0,i])
plt.show()


