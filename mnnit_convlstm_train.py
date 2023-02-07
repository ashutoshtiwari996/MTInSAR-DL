#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 18:42:23 2023

@author: ashutosh
"""
"""
Created on Mon Jan 23 19:04:56 2023

#Hare Krishna

@author: RadheShyam
"""

'''
# A program for selecting PS pixels in multi-temporal InSAR

# Input: 
    1. A stack of interferograms in .mat format (exported from matlab) or in tiff format 
    # (Ifgs exported from ESA SNAP software) 
    # The interferograms initially of size (image rows, image columns, #image ifgs) 
    # are first divided into image patches of 100 by 100 pixels 
    # and then fed to the network
    2. PS labelled map with dimensions image rows by image columns 

# Output: A map with labels 0 and 1, 0 denoting non-PS pixels, and 1 denoting PS pixels 

'''

"""
Packages required

numpy scipy mat73 rasterio hdf5storage matplotlib patchify 

"""


import rasterio as rs
import numpy as np
import scipy.io as spio
import pandas
import hdf5storage as hs
import mat73
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import hdf5storage as hs
import mat73
from patchify import patchify, unpatchify
from PIL import Image


fpath='/home/ashutosh/JSG29-DS and ML in Geodesy/Datasets/MNNIT'

ppath='/home/ashutosh/JSG29-DS and ML in Geodesy/Datasets/MNNIT/StaMPS_export';


dataset=spio.loadmat(fpath+'/MTInSAR_training_dataset.mat')

X=dataset['X']

y=dataset['y']

#Hare Krishna

#crop input to modulo size


# X= X.reshape(X.shape[1], X.shape[2], X.shape[0])

Xrow=X.shape[0]-X.shape[0]%100
Xcol=X.shape[1]-X.shape[1]%100


X1=X[0:Xrow, 0:Xcol]

y1=y[0:Xrow, 0:Xcol]


img_row, img_col=100, 100


hk=patchify(X1, (img_row,img_col,X1.shape[2]), step=100)


gn=patchify(y1, (img_row,img_col), step=100)

# gn=np.split(X, 456, axis=0)

patches=hk

print(patches.shape)

patchesy=gn

# a = np.arange(patches.shape[0]*patches.shape[1]*patches.shape[3]*patches.shape[4]*patches.shape[5])
# a = np.reshape(a, (patches.shape[0]*patches.shape[1],patches.shape[3],patches.shape[4],patches.shape[5]))

patch=np.zeros((patches.shape[0]*patches.shape[1],patches.shape[3],patches.shape[4],patches.shape[5]))

# patch=np.float64(patch)

ctr=0
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch[ctr,:,:,:] = patches[i, j, 0]
        ctr=ctr+1
        
        
ele=np.count_nonzero(patch)

patchy = np.zeros((patchesy.shape[0]*patchesy.shape[1],patchesy.shape[2],patchesy.shape[3]))


ctr2=0
for i in range(patchesy.shape[0]):
    for j in range(patchesy.shape[1]):
        patchy[ctr2,:,:] = patchesy[i, j, :,:]
        ctr2=ctr2+1
        # patch = Image.fromarray(patch)
        # num = i * patches.shape[1] + j
        # patch.save(f"patch_{num}.jpg")
        

img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, X1.shape[2])


Xdiv=patch

ydiv=patchy

# Each image's dimension is 100 by 100


# In[ ]:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Xdiv, ydiv, test_size = 0.2, random_state=0)

#Jai RadheShyam

#resizing training and test datasets to size (#samples, #timesteps, img_rows, img_cols,#bands)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[3], img_rows, img_cols,1)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[3], img_rows, img_cols,1)

y_train = y_train.reshape(y_train.shape[0], img_rows, img_cols, 1)

y_test = y_test.reshape(y_test.shape[0], img_rows, img_cols, 1)

X_full=Xdiv.reshape(Xdiv.shape[0], Xdiv.shape[3], img_rows, img_cols,1)

del dataset # dataset2, X,y


#Jai RadheShyam
#Jai RadheShyam

#Hare Krishna

import tensorflow as tf
import numpy as np
import pandas
import scipy.io as spio
from keras.utils import np_utils
import keras.backend as K
from itertools import product


def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives+K.epsilon())
        return recall


    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#Jai RadheShyam

# def f1(y_true, y_pred):
#     y_pred = K.round(y_pred)
#     tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
#     tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
#     fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
#     fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

#     p = tp / (tp + fp + K.epsilon())
#     r = tp / (tp + fn + K.epsilon())

#     f1 = 2*p*r / (p+r+K.epsilon())
#     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
#     return K.mean(f1)


# def f1_loss(y_true, y_pred):
    
#     tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
#     tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
#     fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
#     fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

#     p = tp / (tp + fp + K.epsilon())
#     r = tp / (tp + fn + K.epsilon())

#     f1 = 1*p*r / (p+r+K.epsilon())
#     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
#     return 1 - K.mean(f1)

#Jai RadheShyam
    
import tensorflow as tf
#print("tensorflow version:", tf.VERSION)
print("tensorflow keras version:", tf.keras.__version__)


def mcor(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())


def precision(y_true, y_pred):
    """ Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return 2*((precision * recall) / (precision+recall + K.epsilon()))


# import keras.backend.tensorflow_backend as tfb

POS_WEIGHT = 200  # multiplier for positive targets, needs to be tuned

def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)


#Jai RadheShyam

#importing libraries for the DL architecture

# from keras.callbacks import LearningRateScheduler
# from keras.utils.training_utils im


# import multi_gpu_model
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import keras.backend as K


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#Jai RadheShyam

#Jai RadheShyam
from keras.models import Sequential
from keras.layers.convolutional import Conv3D

from keras.layers import LSTM
from keras.layers import ConvLSTM2D
from keras.layers import BatchNormalization

# from keras.layers.convolutional_recurrent import ConvLSTM2D
# from keras.layers.normalization import BatchNormalization
from keras.layers import AveragePooling3D, Reshape, Activation, Flatten, Dense

#importing necessary layers
    
from keras import backend as K


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import tensorflow as tf

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

#Jai RadheShyam

#Hare Krishna

batch_size=16
clstm_iss=Sequential()

clstm_iss.add(ConvLSTM2D(filters=16, kernel_size=(3,3), input_shape=(None,100,100,1), padding='same', return_sequences=True, activation='relu'))

# clstm_iss.add(Dropout(0.3))

clstm_iss.add(BatchNormalization())

# clstm_iss.add(ConvLSTM2D(filters=64, kernel_size=(5,5), padding='same', return_sequences=True, activation='relu'))
# clstm_iss.add(BatchNormalization())

clstm_iss.add(ConvLSTM2D(filters=16, kernel_size=(5,5), padding='same', return_sequences=False, activation='relu'))
# clstm_iss.add(Dropout(0.3))

clstm_iss.add(BatchNormalization())

clstm_iss.add(Conv2D(filters=32, kernel_size=(7,7), padding='same', activation='relu'))
# clstm_iss.add(Dropout(0.7))

clstm_iss.add(Dense(1, activation='softmax'))

#compiling strategies

#classifier.compile(loss='binary_crossentropy',optimizer='adadelta', metrics=['accuracy'])

#convlstm1.compile(loss=loss,optimizer='adadelta', metrics=['accuracy'])


clstm_iss.summary()


# In[ ]:


#Jai RadheShyam

from keras.optimizers import SGD, Adam
opt1 = SGD(learning_rate=0.0001, decay=1e-4, momentum=0.9, nesterov=True)

opt2 = Adam(learning_rate=0.01, decay=1e-4)

opt3 = Adam(learning_rate=0.01)


#cnn1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#
#cnn1.compile(optimizer=opt3, loss=f1_loss, metrics=[f1_new])

#Jai RadheShyam

clstm_iss.compile(optimizer=opt1, loss=f1_loss, metrics=['accuracy', f1])

#clstm_iss.compile(optimizer=opt3, loss=f1_loss, metrics=['accuracy'])

#cnn1.compile(optimizer='rmsprop', loss=weighted_binary_crossentropy, metrics=['accuracy', mcor, precision, recall, f1score])


from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=5)

#Jai RadheShyam


#Hare Krishna

clstm_iss.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=[X_test, y_test], callbacks=[early_stopping_monitor])

radhe=clstm_iss.predict(X_test)

ypred_test=radhe

ypred_full=clstm_iss.predict(X_full)

ypred_test= np.where(ypred_test > 0.5, 1, 0)

ypred_full= np.where(ypred_full > 0.5, 1, 0)

shyam=clstm_iss.evaluate(X_test,y_test)

import scipy.io as spio

spio.savemat('radheshyam_convlstm_MNNIT_recall.mat', dict(ypred_test=ypred_test, y_train=y_train, X_test=X_test))


from matplotlib import pyplot as plt
plt.imshow(y, interpolation='nearest')
plt.show()



from keras.models import model_from_json

# serialize classifier to JSON
classifier_json = clstm_iss.to_json()
with open("radheshyam_convlstm_MNNIT_recall.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5
classifier_json.save_weights("radheshyam_convlstm_MNNIT_recall.h5")
print("Saved classifier to disk")

#Jai RadheShyam, now train from loaded model

# load json and create classifier
json_file = open('radheshyam_convlstm_MNNIT_recall.json', 'r')
loaded_classifier_json = json_file.read()
json_file.close()
loaded_classifier = model_from_json(loaded_classifier_json)
# load weights into new classifier
loaded_classifier.load_weights("radheshyam_convlstm_MNNIT_recall.h5")
print("Loaded classifier from disk")


'''
The program gives a map of PS and non-PS pixels 

'''

