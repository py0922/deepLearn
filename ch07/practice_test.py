#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:38:49 2019

@author: Emma
"""

import pandas as pd
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.layers import Conv2D, Flatten,Dropout,Dense
from tensorflow.python.keras.models import Sequential


img_rows,img_cols=28,28
num_classes=10

def data_pre(raw_data):
    y_out=keras.utils.to_categorical(raw_data.label,num_classes)
    num_images=raw_data.shape[0]
    x_array=raw_data.values[:,1:]
    x_shaped_array=x_array.reshape(num_images,img_rows,img_cols,1)
    x_out=x_shaped_array/255
    return x_out,y_out

train_file='train.csv'
raw_data=pd.read_csv('train.csv')

x,y=data_pre(raw_data)

model=Sequential()
model.add(Conv2D(20,kernel_size=(3,3),activation='relu',input_shape=(img_rows,img_cols,1)))
model.add(Conv2D(20,kernel_size=(3,3),strides=2,activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
model.fit(x,y,batch_size=128,epochs=3,validation_split=0.2)