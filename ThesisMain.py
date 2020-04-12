# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 23:11:09 2020

@author: julio
"""
import os
import librosa
from librosa import display 
import matplotlib.pyplot as plt
import numpy as np
import keras
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.utils import np_utils
from sklearn.preprocessing import scale
import CreateModel as CM
import GetEvaluationData as GED
import ConvertGenresToLabels as CGTL
import Normalize as N
import GetSpectrograms as GS    
import GetTestData as GTD
import Equalizer as E
path=os.getcwd()
pathDatasetTraining='C:\\Users\\julio\\Desktop\\Thesis\\Datasets\\TrainingData'
pathDatasetTest='C:\\Users\\julio\\Desktop\\Thesis\\Datasets\\TestData'
duration=29
offsetEv=0
tryAgain='Yes'
genres=np.asarray(os.listdir(pathDatasetTraining))
if(os.path.exists('model.h5')==True):
    reply=input('There is a model already created and trained, do you want to load it? \n')
    if(reply=='Yes'):
            print('Loading model')
            model=keras.models.load_model('model.h5')
            print('Model loaded')
    if(reply=='No'):
        print('The model will be deleted and a new process will begin')
        os.remove('model.h5')
if(os.path.exists('model.h5')==False):
    print('Collecting data')
    MelSpectrogramsNormalized,labels=GS.GetSpectrograms(pathDatasetTraining,genres,duration=duration,offset=0)
    print('Data collected')
    model=CM.CreateModel(MelSpectrogramsNormalized.shape[1],MelSpectrogramsNormalized.shape[2],MelSpectrogramsNormalized.shape[3])
    print('Model Created')
    print('Training of the model will begin')
    test_data,test_labels=GTD.GetTestData(pathDatasetTest,duration=duration,offset=0)
    test_labels=CGTL.ConvertGenresToLabels(test_labels)
    test_labels=np_utils.to_categorical(test_labels,10)
    results=model.fit(MelSpectrogramsNormalized,labels,batch_size=32,epochs=50,verbose=1,validation_data=(test_data,test_labels)) 
    print('Model Trained')
    if (input('Do you want to see the graphic of the accuracies during the training? \n')=='Yes'):
        print('Graphic of the accuracies during the training:')
        plt.plot(results.history['acc'])
        plt.show()
        print('Graphic of the validation accuracy:')
        plt.plot(results.history['val_acc'])
        plt.show()
    model.save('model.h5')
    print('Model saved')
if (input('Do you want to evaluate the model? \n')=='Yes'):
    if 'test_data' not in locals():
            print('Collecting test data')
            test_data,test_labels=GTD.GetTestData(pathDatasetTest,duration=duration,offset=0)
            test_labels=CGTL.ConvertGenresToLabels(test_labels)
            test_labels=np_utils.to_categorical(test_labels,10)
            print('Test data collected')
    print('Evaluation with the method evaluate:')
    scoreEvaluation=model.evaluate(test_data,test_labels,verbose=1)
    print('The evaluation score is: '+str(scoreEvaluation[1]))
    print('Now we will try with different data (Training data but with an offset of '+str(offsetEv)+' seconds):')
    td,tl=GS.GetSpectrograms(pathDatasetTraining,genres,duration=duration,offset=offsetEv)
    scoreEvaluation2=model.evaluate(td,tl,verbose=1)
    print('The second evaluation score is: '+str(scoreEvaluation2[1]))
if(input('Do you want to select a file and try the automatic equalizer?\n')=='Yes'):
    while tryAgain=='Yes':
        Tk().withdraw()
        filename=askopenfilename(title='Select an audio file')
        audio,fs=librosa.load(filename,duration=duration)
        wholeAudio,fs=librosa.load(filename)
        MelSpectrogram=librosa.feature.melspectrogram(audio,fs,power=2.0)
        MelSpectrogramDb=librosa.power_to_db(MelSpectrogram)    
        MelSpectrogramDb=MelSpectrogramDb.reshape((1,1,MelSpectrogram.shape[0],MelSpectrogram.shape[1]))
        MelSpectrogramDb=N.Normalize(MelSpectrogramDb)
        predGenre=genres[model.predict_classes(MelSpectrogramDb)]
        print('The song belongs in the category '+str(predGenre))
        print('This is the mel-spectrogram of the original audio')
        MS=librosa.feature.melspectrogram(wholeAudio,fs,power=2.0)
        MSDb=librosa.power_to_db(MS)
        librosa.display.specshow(MSDb,x_axis='time',y_axis='mel',sr=fs)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.show()
        audioEq=E.equalizer(wholeAudio,fs,predGenre)
        print('And this is the mel-spectrogram of the equalized audio')
        MSe=librosa.feature.melspectrogram(audioEq,fs,power=2.0)
        MSDbe=librosa.power_to_db(MSe)
        librosa.display.specshow(MSDbe,x_axis='time',y_axis='mel',sr=fs)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.show()
        audioEq=E.equalizer(wholeAudio,fs,predGenre)
        tryAgain=input('Do you want to try again? \n')




