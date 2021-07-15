#data augmenatation
import librosa
import math
import array
import os
import numpy as np
from sklearn.utils import shuffle
import feature_extraction as ft


def extract_audio(directory):
    a = []
    lbl = []  
    for filename in os.listdir(directory):  
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            x, sr = librosa.load(filepath)
            l = ft.queen_info(filepath)
            a.append(x)
            lbl.append(l)
    a = np.asarray(a)
    lbl = np.asarray(lbl)
    return a, lbl
                        


#-------data augmentation-----------------#
def data_augmentation(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks):
        stft_mean = []
        fold1_audios, fold1_labels = extract_audio(fold1_directory)
        fold2_audios, fold2_labels = extract_audio(fold2_directory)
        fold3_audios, fold3_labels = extract_audio(fold3_directory)
        fold4_audios, fold4_labels = extract_audio(fold4_directory)
        audios = np.concatenate((fold1_audios, fold2_audios, fold3_audios, fold4_audios))
        labels = np.concatenate((fold1_labels, fold2_labels, fold3_labels, fold4_labels))
        audios, labels = shuffle(audios, labels, random_state = 0)
        x = int(audios.shape[0]*0.5)
        audios = audios[:x]
        labels = labels[:x]
        print(audios.shape)
        for x in audios:
            SNR = 20
            RMS_s=math.sqrt(np.mean(x**2))
            RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
            STD_n = RMS_n
            noise = np.random.normal(0, STD_n, x.shape)
            a = x + noise
            s = np.abs(librosa.stft(a, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'))
            s_mean = ft.mean(s, n_chunks)
            stft_mean.append(s_mean)
        stft_mean = np.asarray(stft_mean)
        print (stft_mean.shape, labels.shape)
        return stft_mean, labels
#------------------------------------------#  