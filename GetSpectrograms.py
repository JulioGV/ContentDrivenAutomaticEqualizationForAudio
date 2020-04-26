# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:14:04 2020

@author: julio
"""
##################################################################
# Author: Julio González Villegas                                #     
# email: juliogonzalez9634@gmail.com                             #
#                                                                #
# This script is used to get the mel-spectrograms of the tracks  #
# that are going to be used for the training of the model.       #
#                                                                #
##################################################################

import os
import librosa
import numpy as np
import Normalize as N
import ConvertGenresToLabels as CGTL
from keras.utils import np_utils

def GetSpectrograms(pathDataset,genres,duration,offset):
#    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split() MAYBE NOT NEEDED, TRY
    Ms=[]
    labels=[]
    for filename in os.listdir(pathDataset):
            for filename1 in os.listdir(pathDataset+'\\'+filename):
                audio,fs=librosa.load(pathDataset+'\\'+filename+'\\'+filename1,duration=duration,offset=offset)

                MelSpectrogram=librosa.feature.melspectrogram(audio,fs,power=2.0)
                MelSpectrogramDb=librosa.power_to_db(MelSpectrogram)
                Ms.append(MelSpectrogramDb)
    MelSpectrograms=np.asarray(Ms)
    labels=np.repeat(genres,len(MelSpectrograms)/len(genres))
    MelSpectrograms=MelSpectrograms.reshape(MelSpectrograms.shape[0],1, MelSpectrograms.shape[1], MelSpectrograms.shape[2])
    MelSpectrograms=MelSpectrograms.astype('float32')
    MelSpectrogramsNormalized = N.Normalize(MelSpectrograms)
    labels=np.repeat(genres,len(MelSpectrograms)/len(genres))
    labels=CGTL.ConvertGenresToLabels(labels)
    labels=np_utils.to_categorical(labels,10)
    return MelSpectrogramsNormalized,labels 
        
