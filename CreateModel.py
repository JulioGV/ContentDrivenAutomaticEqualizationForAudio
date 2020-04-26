# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 00:03:47 2020

@author: julio
"""
##################################################################
# Author: Julio González Villegas                                #     
# email: juliogonzalez9634@gmail.com                             #
#                                                                #
# Script for the creation of the model, the input dimensions     #
# are passed as parameters.                                      #
##################################################################


from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D


def CreateModel(depth,width,height):
    model=Sequential()
    model.add(Convolution2D(32, kernel_size=5, strides=2, activation='relu', input_shape=(depth,width,height),data_format='channels_first'))
    model.add(Convolution2D(64, kernel_size=3, strides=1, activation='relu'))       
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model