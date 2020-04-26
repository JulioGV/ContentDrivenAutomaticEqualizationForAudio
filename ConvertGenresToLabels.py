# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:57:48 2020

@author: julio
"""
##################################################################
# Author: Julio González Villegas                                #     
# email: juliogonzalez9634@gmail.com                             #
#                                                                #
# Auxiliary function to convert the names of the genres into     #
# numerical labels                                               #
#                                                                #
##################################################################
import numpy as np

def ConvertGenresToLabels(g):
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    d={}
    g=np.asarray(g)
    for i in range(len(genres)):
        d[genres[i]]=i
    labels=[]
    for i in range(len(g)):
        labels.append(d.get(g[i]))
    return labels
          
    
