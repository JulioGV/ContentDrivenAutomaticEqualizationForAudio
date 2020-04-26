# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:03:47 2020

@author: julio
"""
##################################################################
# Author: Julio González Villegas                                #     
# email: juliogonzalez9634@gmail.com                             #
#                                                                #
# Auxiliary function to normalize a variable. Used to normalize  #
# the mel-spectrogram values.                                    #
#                                                                #
##################################################################
import numpy as np
def Normalize(data ) :
    shape = data.shape
    data = np.reshape( data , (-1 , ) )
    maximum = np.max( data )
    minimum = np.min( data )
    normalized_values = list()
    for x in data:
        x_normalized = ( x - minimum ) / ( maximum - minimum )
        normalized_values.append( x_normalized )
    n_array = np.array( normalized_values )
    normalized_array=np.reshape(n_array,shape)
    return normalized_array