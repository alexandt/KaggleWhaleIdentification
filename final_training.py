#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:02:36 2018

@author: Trystan
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

num_classes = len(y_cat.toarray()[0])
epochs = 9

input_shape = (256,256)


#convlution nerual net, keras implementation

model = Sequential()
model.add(Conv2D(Conv2D(48, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape),input_shape=input_shape))