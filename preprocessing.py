#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:02:48 2018

@author: Trystan Alexander
"""
import random
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from keras.preprocessing.image import (
    random_rotation, random_shear, random_zoom,
    img_to_array)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D

from keras.utils import np_utils
import os


train_df = pd.read_csv('./train.csv')

#These methods from Lex Toumbourou's published notebook on Kaggle
def plot_images_for_filenames(filenames, labels, rows=4):
    #creates a list of image plots
    imgs = [plt.imread(f'./train/{filename}') for filename in filenames]
    #plots the created list of images
    return plot_images(imgs, labels, rows)

def plot_images(imgs, labels, rows=4):
    # Set figure to 13 inches x 8 inches
    figure = plt.figure(figsize=(13, 8))

    cols = len(imgs) // rows + 1

    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        if labels:
            subplot.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i], cmap='gray')
        
def save_images(imgs, img_id, folder, df, label, add_to_df=True):
    for i in range(len(imgs)):
        im_id = img_id[:-4]
        c = imgs[i]
        i_name = '%s_aug%.2d.jpg'%(im_id, i%len(imgs))
        fname = './%s/%s'% (folder, i_name)
        plt.imsave(fname, c, cmap='gray')
        new_list = [i_name, label]
        new_dict = {'Image':[i_name], 'Id':[label]}
        #, columns=['Image','Id']
        addition = pd.DataFrame(new_dict)
        if add_to_df:
            df = df.append(addition, ignore_index=True)
    
        
def resize(images, folder_start, folder_dest):
    c = 0
    for image in images:
        path = f'./%s/%s'%(folder_start, image)
        img = Image.open(path).convert(mode='L')
        img = img.resize((256,256))
        img_arr = img_to_array(img)
        if 1 in img_arr.shape:
            img_arr = np.squeeze(img_arr)
        plt.imsave('./%s/%s'%(folder_dest, image), img_arr, cmap='gray')
        c+= 1
        print("%d Images Resized of %d"%(c, len(images)))

def augmentation_pipeline(img_arr):
    img_arr = random_rotation(img_arr, 18, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    img_arr = random_zoom(img_arr, zoom_range=(0.9, 2.0), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    return img_arr




def augment_images(image_ids, folder, image_labels, df):
    c = 0
    for image in image_ids:
        path = f'./train/%s'%image
        img = Image.open(path).convert(mode='L')
        img = img.resize((256,256))
        img_arr = img_to_array(img)
        imgs = [augmentation_pipeline(img_arr) / 255 for _ in range(2)]
        if 1 in imgs[0].shape:
            imgs = np.squeeze(imgs)
        save_images(imgs, image, folder, df, image_labels[c])
        c+= 1
        print("%d Images Augmented of %d"%(c, len(image_ids)))




practice = train_df.head(5)
#augment_images(practice['Image'], 'augmented_train_practice', practice['Id'])

folder = 'at_least_20'

count = Counter(train_df['Id'])
need_augment = []
augment_labels = []
for v in count:
    if count[v] < 2:
        imgs = train_df.loc[train_df['Id']== v, ['Image']].values
        for img in imgs:
            need_augment.append(img[0])
            augment_labels.append(v)
        


    
#augment_images(need_augment, folder, augment_labels, train_df)

test_images = []
for filename in os.listdir('./test'):    
    if filename.endswith(".jpg"):
        test_images.append(filename)

#resize(test_images, './test', './test_resized')

train_images = train_df['Image']

le = LabelEncoder()
categories = le.fit_transform(train_df['Id'].iloc[:-2000])
test_categories = le.fit_transform(train_df['Id'].iloc[-2000:])
Y_train = np_utils.to_categorical(categories, 4251)
Y_test = np_utils.to_categorical(test_categories, 4251)


#X_train = np.array([np.squeeze(img_to_array(Image.open('./at_least_20/%s'%img).convert(mode='L'))) for img in train_images])
#X_test = np.array([np.squeeze(img_to_array(Image.open('./test_resized/%s'%img).convert(mode='L'))) for img in test_images])
X_train = np.array([np.squeeze(img_to_array(Image.open('./at_least_20/%s'%train_images[i]).convert(mode='L'))) for i in range(len(train_images)-2000)])
X_test = np.array([np.squeeze(img_to_array(Image.open('./at_least_20/%s'%img).convert(mode='L'))) for img in train_images[-2000:]])


X_train = X_train.reshape(X_train.shape[0], 256, 256, 1)

X_test = X_test.reshape(X_test.shape[0], 256, 256, 1)


X_train = X_train.astype('float32')
X_train /= 255

X_test = X_test.astype('float32')
X_test /= 255

#outfile = open('x_train', 'wb')
#pickle.dump([X_train], outfile)
#outfile.close()
#
#outfile = open('x_test', 'wb')
#pickle.dump([X_test], outfile)
#outfile.close()
#
#outfile = open('y_train', 'wb')
#pickle.dump([Y_train], outfile)
#outfile.close()


input_shape = (256,256,1)
#
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4251, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#
model.fit(X_train, Y_train, 
         batch_size=32, epochs=1, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)

print(score)

model2 = Sequential()
model2.add(Conv2D(48, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model2.add(Conv2D(48, (3, 3), activation='sigmoid'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(48, (5, 5), activation='sigmoid'))
model2.add(MaxPooling2D(pool_size=(3, 3)))
model2.add(Dropout(0.33))
model2.add(Flatten())
model2.add(Dense(36, activation='sigmoid'))
model2.add(Dropout(0.33))
model2.add(Dense(36, activation='sigmoid'))
model2.add(Dense(4251, activation='softmax'))

model2.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])

#model2.fit(X_train, Y_train, 
#         batch_size=48, epochs=10, verbose=1)
#        
