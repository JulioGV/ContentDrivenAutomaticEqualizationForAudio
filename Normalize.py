# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:03:47 2020

@author: julio
"""
import numpy as np
def Normalize(data ) :
    # Store the data's original shape
    shape = data.shape
    # Flatten the data to 1 dimension
    data = np.reshape( data , (-1 , ) )
    # Find minimum and maximum
    maximum = np.max( data )
    minimum = np.min( data )
    # Create a new array for storing normalized values
    normalized_values = list()
    # Iterate through every value in data
    for x in data:
        # Normalize
        x_normalized = ( x - minimum ) / ( maximum - minimum )
        # Append it in the array
        normalized_values.append( x_normalized )
    # Convert to numpy array
    n_array = np.array( normalized_values )
    # Reshape the array to its original shape and return it.
    return np.reshape( n_array , shape )