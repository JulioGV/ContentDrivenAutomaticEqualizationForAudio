# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 01:08:31 2020

@author: julio
"""
##################################################################
# Author: Julio González Villegas                                #     
# email: juliogonzalez9634@gmail.com                             #
#                                                                #
# Auxiliary script to plot the fft of an audio                   #      
#                                                                #
##################################################################
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
def fft_plot(audio,fs,title):
    n= len(audio)
    y=sp.fft(audio)
    f = fs*np.arange((n/2))/n
    fig,ax=plt.subplots()
    ax.plot(f,2.0/n*np.abs(y[:n//2]))
    plt.grid()
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Amplitude')
    plt.title(title)
    return plt.show()