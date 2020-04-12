# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 18:10:39 2020

@author: julio
"""
import scipy.signal as signal
from scipy.signal import butter,lfilter
import numpy as np
import matplotlib.pyplot as plt
import librosa

import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def equalizer(audio,fs,genre):
    fs=22050
    gains=getGains(genre)
    b1,a1= bandpass_filter(audio,32,fs,3)
    band1=lfilter(b1, a1, audio)
    band1=band1*10**(gains[0]/20)
    b2,a2 = bandpass_filter(audio, 62, fs,3)
    band2=lfilter(b2,a2,audio)
    band2=band2*10**(gains[1]/20)
    b3,a3 = bandpass_filter(audio, 125, fs, order=3)
    band3=lfilter(b3,a3,audio)
    band3=band3*10**(gains[2]/20)
    b4,a4=bandpass_filter(audio,250,fs,order=3)
    band4=lfilter(b4,a4,audio)
    band4=band4*10**(gains[3]/20)
    b5,a5 = bandpass_filter(audio,500,fs,5)
    band5=lfilter(b5,a5,audio)
    band5=band5*10**(gains[4]/20)
    b6,a6 = bandpass_filter(audio,1000,fs,5)
    band6=lfilter(b6,a6,audio)
    band6=band6*10**(gains[5]/20)
    b7,a7 = bandpass_filter(audio,2000,fs,5)
    band7=lfilter(b7,a7,audio)
    band7=band7*10**(gains[6]/20)
    b8,a8 = bandpass_filter(audio,4000,fs,5)
    band8=lfilter(b8,a8,audio)
    band8=band8*10**(gains[7]/20)
    b9,a9 = shelving_filter(audio,4000,fs,5,highlow='high')
    band9=lfilter(b9,a9,audio)
    band9=band9*10**(gains[8]/20)
    equalizedSignal=band1+band2+band3+band4+band5+band6+band7+band8+band9
    showFreqResponse(b1,a1,gains[0],'b',genre)
    showFreqResponse(b2,a2,gains[1],'g',genre)
    showFreqResponse(b3,a3,gains[2],'r',genre)
    showFreqResponse(b4,a4,gains[3],'c',genre)
    showFreqResponse(b5,a5,gains[4],'m',genre)
    showFreqResponse(b6,a6,gains[5],'y',genre)
    showFreqResponse(b7,a7,gains[6],'k',genre)
    showFreqResponse(b8,a8,gains[7],'#ff9990',genre)
    showFreqResponse(b9,a9,gains[8],'b--',genre)
    plt.show()
    return equalizedSignal
def getGains(genre):
        if(genre=='blues'):
            gains=-1,0,2,1,0,0,0,0,-1            
        
        if(genre=='classical'):
            gains=0,6,6,3,0,0,0,0,2
        
        if(genre=='country'):
            gains=-1,0,0,2,2,0,0,0,3
        
        if(genre=='disco'):
            gains=-1,4,5,1,-1,-1,0,0,4
        
        if(genre=='hiphop'):
            gains=-1,0,2,2,-1,-1,0,0,4
        
        if(genre=='jazz'):
            gains=0,0,0,3,3,3,0,2,4
        
        if(genre=='metal'):
            gains=-4,0,0,0,0,0,3,0,3
        
        if(genre=='pop'):
            gains=-1,2,3,4,6,3,0,-1,-2
        
        if(genre=='reggae'):
            gains=-1,0,0,-3,0,3,4,0,3
        
        if(genre=='rock'):
            gains=-1,1,2,3,-1,-1,0,0,4
            
        if(genre=='neutral'):
            gains=0,0,0,0,0,0,0,0,0
        
        return gains

def createFilter(centralFreq,fs,order):
    nyq=0.5*fs
    lowcut=centralFreq-centralFreq/2
    highcut=centralFreq+centralFreq/2
    if(lowcut==0):
        lowcut=lowcut+0.0001
    low=lowcut/nyq
    high=highcut/nyq
    b,a=butter(order,[low,high],btype='bandpass')
    return b,a
def showFreqResponse(b,a,g,color,genre):
    fs=22050
    w, h = signal.freqz(b, a, worN=8000)
    plt.subplot(2, 1, 1)
    if (g==0):
        plt.plot(0.5*fs*w/np.pi,np.abs(h),color)
    if(g!=0):
        plt.plot(0.5*fs*w/np.pi, np.abs(h)*g, color)
    plt.xlim(0, 0.5*fs)
    plt.xscale('log')
    plt.title(genre+ " BankFilter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    return None
def showSpectrogram(audio,fs):
    N = 512
    f, t, Sxx = signal.spectrogram(audio, fs,window = signal.blackman(N),nfft=N)
    plt.figure()
    plt.pcolormesh(t, f,10*np.log10(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [seg]')
    plt.title('Spectrogram',size=16);
    plt.show()
def displayFFT(audio):
   plt.magnitude_spectrum(audio,Fs=22050)
   plt.show()
def bandpass_filter(data, centralFreq, fs, order):
    nyq = 0.5 * fs
    lowcut=centralFreq-centralFreq/2
    highcut=centralFreq+centralFreq/2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b,a
def shelving_filter(data,cutoff,fs,order,highlow):
    nyq=0.5*fs
    cutoff=cutoff/nyq
    b,a=butter(order,[cutoff],btype=highlow)
    return b,a
def showEqCurve(genre):
    gains=getGains(genre)
    audio=0
    fs=22100
    b1,a1= bandpass_filter(audio,32,fs,3)
    b2,a2 = bandpass_filter(audio, 62, fs,3)
    b3,a3 = bandpass_filter(audio, 125, fs, order=3)
    b4,a4=bandpass_filter(audio,250,fs,order=5)
    b5,a5 = bandpass_filter(audio,500,fs,5)
    b6,a6 = bandpass_filter(audio,1000,fs,5)
    b7,a7 = bandpass_filter(audio,2000,fs,5)
    b8,a8 = bandpass_filter(audio,4000,fs,5)
    b9,a9 = shelving_filter(audio,4000,fs,5,highlow='high')
    showFreqResponse(b1,a1,gains[0])
    showFreqResponse(b2,a2,gains[1])
    showFreqResponse(b3,a3,gains[2])
    showFreqResponse(b4,a4,gains[3])
    showFreqResponse(b5,a5,gains[4])
    showFreqResponse(b6,a6,gains[5])
    showFreqResponse(b7,a7,gains[6])
    showFreqResponse(b8,a8,gains[7])
    showFreqResponse(b9,a9,gains[8])
#------------MAIN FOR TESTS----------------

path='C:\\Users\\julio\\Desktop\\Thesis\\Datasets\\TrainingData\\blues\\blues.00034.wav'
audio,fs=librosa.load(path)
equalizedSignal=equalizer(audio,fs,'jazz')
equalizedSignal=equalizer(audio,fs,'hiphop')
equalizedSignal=equalizer(audio,fs,'classical')
equalizedSignal=equalizer(audio,fs,'rock')


##print('SPECTRUM OF THE ORIGINAL SIGNAL')
##showSpectrogram(audio,fs)
##print('FFT of the original signal')
##displayFFT(audio)
##print('Spectrum of the equalized Signal')
##showSpectrogram(equalizedSignal,fs)
##print('FFT of the equalized signal')
##displayFFT(equalizedSignal)
#
##print('SPECTRUMS AND FFTS OF THE BANDS:')
##print('FIRST BAND')
##showSpectrogram(band1,fs)
##displayFFT(band1)
##print('SECOND BAND')
##showSpectrogram(band2,fs)
##displayFFT(band2)
##print('THIRD BAND')
##showSpectrogram(band3,fs)
##displayFFT(band3)
##print('FOURTH BAND')
##showSpectrogram(band4,fs)
##displayFFT(band4)
##print('FIFTH BAND')
##showSpectrogram(band5,fs)
##displayFFT(band5)
##print('SIXTH BAND')
##showSpectrogram(band6,fs)
##displayFFT(band6)
##print('SEVENTH BAND')
##showSpectrogram(band7,fs)
##displayFFT(band7)
##print('EIGHT BAND')
##showSpectrogram(band8,fs)
##displayFFT(band8)
##print('NINTH BAND')
##showSpectrogram(band9,fs)
##displayFFT(band9)
##
