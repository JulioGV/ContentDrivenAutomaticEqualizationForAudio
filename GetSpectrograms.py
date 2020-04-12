# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:14:04 2020

@author: julio
"""
import os
import librosa
from librosa import display 
import matplotlib.pyplot as plt
import numpy as np
import GetTempo as GT
import Normalize as N
import ConvertGenresToLabels as CGTL
from keras.utils import np_utils

def GetSpectrograms(pathDataset,genres,duration,offset):
#pathDataset='C:\\Users\\julio\\Desktop\\Thesis\\Datasets\\genres'
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    Ms=[]
    labels=[]
#    tempos=[]
    #for j in (0,3):
    for filename in os.listdir(pathDataset):
#            print('Start with '+filename)
    #         for filename1 in os.listdir(pathDataset+'\\'+filename):
            for filename1 in os.listdir(pathDataset+'\\'+filename):
                audio,fs=librosa.load(pathDataset+'\\'+filename+'\\'+filename1,duration=duration,offset=offset)
#                    print('El tempo es: ')
#                tempo=GT.GetTempo(audio,fs)
#                    print(tempo)
                MelSpectrogram=librosa.feature.melspectrogram(audio,fs,power=2.0)
                MelSpectrogramDb=librosa.power_to_db(MelSpectrogram)
    #            librosa.display.specshow(MelSpectrogramDb,x_axis='time',y_axis='mel',sr=fs)
    #            plt.colorbar(format='%+2.0f dB')
    #            plt.title('Mel-frequency spectrogram')
    #            plt.tight_layout()
    #            plt.show()
    #            Ms=np.append((MS,MelSpectrogramDb),0)
                Ms.append(MelSpectrogramDb)
#                tempos.append(tempo)
        #            mfcc=librosa.feature.mfcc(audio,sr=fs,n_mfcc=n_mfcc).T
        #            mfcc_scaled=scaler.fit_transform(mfcc)
        #            mfcc_coeficients_scaled=np.vstack((mfcc_coeficients_scaled,mfcc_scaled))
    MelSpectrograms=np.asarray(Ms)
    labels=np.repeat(genres,len(MelSpectrograms)/len(genres))
    MelSpectrograms=MelSpectrograms.reshape(MelSpectrograms.shape[0],1, MelSpectrograms.shape[1], MelSpectrograms.shape[2])
    MelSpectrograms=MelSpectrograms.astype('float32')
    MelSpectrogramsNormalized = N.Normalize(MelSpectrograms)
    labels=np.repeat(genres,len(MelSpectrograms)/len(genres))
    labels=CGTL.ConvertGenresToLabels(labels)
    labels=np_utils.to_categorical(labels,10)
    
    return MelSpectrogramsNormalized,labels #ADD THE TEMPO
        
