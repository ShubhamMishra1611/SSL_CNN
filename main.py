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

DEBUG = True

print(tf.__version__)

from extract_audio import extractAudiodata
from utils import sorted_nicely
from calcs import featureExtractor2, computeSI
from model import get_model
import logging
logging.basicConfig(filename='ssl_log.log', level=logging.DEBUG, format='%(asctime)s %(message)s')


# get the list of audio files and sort them
logging.info('Getting the list of audio files and sorting them...')
audiofiles = [str(file) for file in Path().glob('SA*.wav')]
audiofiles = sorted_nicely(audiofiles)


if DEBUG:
    print("Selecting Random files from the audio files...")
    print(random.choices(audiofiles, k=10))

# Exctracting audio for the duration of 500 ms with respective labels
sr = 16000
duration = 0.5

audioMatrix, sourceIDs, labels30, labels10 = extractAudiodata(audiofiles, sr*duration)

logging.info(f'audioMatrix has shape {audioMatrix.shape} where {audioMatrix.shape[0]} is length in samples, {audioMatrix.shape[1]} are the microphones and {audioMatrix.shape[2]} are the number of setences')

print('audioMatrix has shape {} where {} is length in samples, {} are the microphones and {} are the number of setences'.format(audioMatrix.shape, audioMatrix.shape[0], audioMatrix.shape[1], audioMatrix.shape[2]))

logging.info(f'sourceIDs is an array of length {sourceIDs.shape[0]}, one for each sentence')
print('labels10 is an array of length {}, one for each sentence\n'.format(labels10.shape[0]))

logging.info(f'plotting the audio signals...')
plt.figure(figsize=(15, 10))
time_axis = np.arange(0,sr*duration) / sr
for i in range(0, 10):
  plt.subplot(5, 2, i+1)
  plt.plot(time_axis, audioMatrix[:,0,i])

gamma = featureExtractor2(audioMatrix[:,:,0])


# Get list of audio files in the train dataset
audiofiles_train = [str(file) for file in Path().glob('SA*.wav')]
audiofiles_train = sorted_nicely(audiofiles_train)

sr = 16000
duration = 0.5

# Extract audio and labels
trainAudioMatrix, sourceIDs, labels30, labels10 = extractAudiodata(audiofiles_train, sr*duration)

# Init gamma
datapointsNum = trainAudioMatrix.shape[-1]
Gamma_container = np.zeros([datapointsNum, 14, 511, 10])

# Extract SI features
for i in tqdm(range(0, datapointsNum)):
  Gamma_container[i,:,:,:] = featureExtractor2(trainAudioMatrix[:,:,i])

# Casting labels as integers
intLabels10 = np.array(labels10, dtype='uint8')
intLabels30 = np.array(labels30, dtype='uint8')


# Get list of audio files in the validation dataset
audiofiles_val = [str(file) for file in Path().glob('SA*.wav')]
audiofiles_val = sorted_nicely(audiofiles_val)

sr = 16000
duration = 0.5

# Extract audio and labels
valAudioMatrix, sourceIDs, valLabels30, valLabels10 = extractAudiodata(audiofiles_val, sr*duration)

# Init gamma
datapointsNum = valAudioMatrix.shape[-1]
Gamma_container_val = np.zeros([datapointsNum, 14, 511, 10])

# Extract SI features
for i in tqdm(range(0, datapointsNum)):
  Gamma_container_val[i,:,:,:] = featureExtractor2(valAudioMatrix[:,:,i])

# Casting labels as integers
intValLabels10 = np.array(valLabels10, dtype='uint8')
intValLabels30 = np.array(valLabels30, dtype='uint8')

logging.info('Getting the model...')
model = get_model()

logging.info('Compiling the model...')
learning_rate = 1e-5
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
)

logging.info('defining callbacks...')
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


callbacks = [
  model_checkpoint_callback,
  tf.keras.callbacks.EarlyStopping(patience=25, monitor='accuracy', mode='max'),
]

(Gamma_container, intLabels10) = shuffle(Gamma_container, intLabels10)
(Gamma_container_val, intValLabels10) = shuffle(Gamma_container_val, intValLabels10)


history = model.fit(Gamma_container, intLabels10, validation_data=(Gamma_container_val, intValLabels10),  epochs=400, callbacks=callbacks)

logging.info('saving the model...')
model.save('model')

plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(1,2,2)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')





