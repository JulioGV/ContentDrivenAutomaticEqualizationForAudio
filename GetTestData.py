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
# that are going to be used for the testing of the model.        #
#                                                                #
##################################################################
import os
import librosa
import numpy as np
import Normalize as N
def GetTestData(pathDatasetTest,duration,offset):
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    Ms=[]
    labels=[]
    for filename in os.listdir(pathDatasetTest):
            for filename1 in os.listdir(pathDatasetTest+'\\'+filename):
                audio,fs=librosa.load(pathDatasetTest+'\\'+filename+'\\'+filename1,duration=duration,offset=offset)
                MelSpectrogram=librosa.feature.melspectrogram(audio,fs,power=2.0)
                MelSpectrogramDb=librosa.power_to_db(MelSpectrogram)
                Ms.append(MelSpectrogramDb)
    MelSpectrograms=np.asarray(Ms)
    MelSpectrograms=MelSpectrograms.reshape(MelSpectrograms.shape[0],1, MelSpectrograms.shape[1], MelSpectrograms.shape[2])
    MelSpectrograms=MelSpectrograms.astype('float32')
    MelSpectrogramsNormalized = N.Normalize(MelSpectrograms)
    labels=np.repeat(genres,len(MelSpectrograms)/len(genres))
    
    return MelSpectrogramsNormalized,labels 