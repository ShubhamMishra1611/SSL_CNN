import numpy as np
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

# Sound Intensity (SI) estimation
def computeSI(X1, X2, X3, freqsMatrix, isOblique=False, X4=0,  d=0.04, rho=1.225, beta=1e6):
  denominator = (freqsMatrix)*d*rho

  if isOblique:
    # Paricle velocity estimation in TF-domain relative to subarrays' directions (pi/4, 3/4pi, -pi/4, -3/4pi)
    X0 = (X1 + X2 + X3) / 3
    V1 = (np.sqrt(2)*1j*(X2 - X1)) / denominator
    V2 = (np.sqrt(2)*1j*(X2 - X3)) / denominator
  else:
    # Paricle velocity estimation in TF-domain relative to principal directions (x and y)
    X0 = (X1 + X2 + X3 + X4) / 4
    V1 = (1j*(X3 - X1)) / denominator
    V2 = (1j*(X4 - X2)) / denominator

  # Witheningh weight
  W = (np.abs(X0)**2 + beta*(np.abs(V1)**2 + np.abs(V2)**2) )**0.5 + np.finfo(np.float32).eps

  # Extracted SI features
  I1 = (np.real(X0*np.conj(V1)) / W).T
  I2 = (np.real(X0*np.conj(V2)) / W).T

  # Grouping features into a matrix
  matrix = np.zeros([I1.shape[0], I1.shape[1], 2])
  matrix[:, :, 0] = I1
  matrix[:, :, 1] = I2

  return matrix

# Compute the SI features for each designated DMA (Differential Microphone Array): principal directions + subarrays
def featureExtractor2(micSignals, d=0.04, rho=1.225, NFFT=1024, sr=16000, beta=1e6):
  # STFT variables
  frame_length = NFFT
  hop_length = int(frame_length/2) # 50% overlap
  num_samples = micSignals[:,0].shape[0] # Length in samples
  num_frames = int(1 + np.floor((num_samples - frame_length) / hop_length)) # Resulting number of frames

  # freqMatrix is a matrix that contains the angular frequency axis repeated on each column
  # It is used for the SI estimation
  freqs = librosa.fft_frequencies(sr=sr, n_fft=NFFT) / (sr/2) * np.pi + np.finfo(np.float32).eps
  freqs = np.reshape(freqs, (len(freqs),1))
  freqsMatrix = np.tile(freqs, (1, num_frames))

  # Init output matrix
  gamma = np.zeros([num_frames, int(NFFT/2 - 1), 10])

  # STFT microphones' signals
  P1 = librosa.stft(micSignals[:,0], n_fft = NFFT, hop_length = hop_length, win_length = frame_length, window='hann', center=False)
  P2 = librosa.stft(micSignals[:,1], n_fft = NFFT, hop_length = hop_length, win_length = frame_length, window='hann', center=False)
  P3 = librosa.stft(micSignals[:,2], n_fft = NFFT, hop_length = hop_length, win_length = frame_length, window='hann', center=False)
  P4 = librosa.stft(micSignals[:,3], n_fft = NFFT, hop_length = hop_length, win_length = frame_length, window='hann', center=False)

  # Principal directions SI extraction
  # First and last frequency bins are excluded because always equal to 0
  gamma[:,:,0:2] = computeSI(P1, P2, P3, freqsMatrix, isOblique=False, X4=P4, d=d, rho=rho, beta=beta)[:,1:int(NFFT/2),0:2]

  # Subarrays SI extraction
  gamma[:,:,2:4]  = computeSI(P4, P1, P2, freqsMatrix, isOblique=True, d=d, rho=rho, beta=beta)[:,1:int(NFFT/2),0:2]
  gamma[:,:,4:6]  = computeSI(P1, P2, P3, freqsMatrix, isOblique=True, d=d, rho=rho, beta=beta)[:,1:int(NFFT/2),0:2]
  gamma[:,:,6:8]  = computeSI(P2, P3, P4, freqsMatrix, isOblique=True, d=d, rho=rho, beta=beta)[:,1:int(NFFT/2),0:2]
  gamma[:,:,8:10] = computeSI(P3, P4, P1, freqsMatrix, isOblique=True, d=d, rho=rho, beta=beta)[:,1:int(NFFT/2),0:2]

  # Gamma is a 14*511*10 matrix where
  #   - 14 is the number of frames
  #   - 511 is the number of frequency bins
  #   - 10 is the number of channels (SI for each DMA)

  return gamma