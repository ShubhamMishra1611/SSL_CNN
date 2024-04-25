# import numpy as np
# import librosa
# from tqdm import tqdm

# # Generate a matrix containing audio data as columns
# # This function assumes the alphanumerical ordering of the filepaths
# def extractAudiodata(filepaths, sample_len):
#   sentencesNum = int(len(filepaths)/4) # Number of sentences
#   sentencesId = np.zeros([sentencesNum, 1]) # Array containing all the sentences' ID
#   labels30 = np.zeros(sentencesId.shape) # Lables for the model with resolution 30°
#   labels10 = np.zeros(sentencesId.shape) # Lables for the model with resolution 10°

#   # Ouput audio matrix N*M*L
#   # N : Length of the audio files in number of samples
#   # M : Number of microphones
#   # L : Number of sentences
#   audioSignals = np.zeros([int(sample_len), 4, sentencesNum])


#   j = 0 # j is indexing sentences, range: [0, sentencesNum - 1]
#   for i in tqdm(range(0, len(filepaths), 4)): # i is indexing samples, range: [0, len(filepath)] with step 4
#     # Selecting the four microphones' input for a given sentence
#     micsABCD = filepaths[i:i+4]

#     for k in range(0, len(micsABCD)): # k is indexing mics, range: [0,3]
#       # Populating the ouptut matrix
#       # Loading k-th microphone's input of the j-th sentence as a column vector
#       audioSignals[:, k, j] = librosa.load(micsABCD[k], sr=None)[0]

#     # Extracting labels and sentence id from the sample relative to the first microphone
#     fileid = micsABCD[0].split('.wav')[-2]
#     sentencesId[j] = fileid.split('-')[-3]
#     labels30[j] = fileid.split('-')[-2]
#     labels10[j] = fileid.split('-')[-1]

#     # Increasing sentence's index
#     j += 1


#   return audioSignals, sentencesId, labels30, labels10

import numpy as np
import librosa
from tqdm import tqdm

def extractAudiodata(filepaths, sample_len):
    sentencesNum = len(filepaths) // 4
    sentencesId = np.zeros((sentencesNum, 1))
    labels30 = np.zeros(sentencesId.shape)
    labels10 = np.zeros(sentencesId.shape)
    audioSignals = np.zeros([int(sample_len), 4, sentencesNum])

    for j, i in enumerate(tqdm(range(0, len(filepaths), 4))):
        micsABCD = filepaths[i:i+4]

        for k, mic_path in enumerate(micsABCD):
            audioSignals[:, k, j] = librosa.load(mic_path, sr=None)[0]

        fileid = mic_path.split('.wav')[0].split('-')
        sentencesId[j] = fileid[-3]
        labels30[j] = fileid[-2]
        labels10[j] = fileid[-1]

    return audioSignals, sentencesId, labels30, labels10
