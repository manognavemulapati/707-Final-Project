from __future__ import division, print_function
import json
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
#import cv2
#import pandas as pd
import numpy as np
#import biosppy
import matplotlib.pyplot as plt
#import wfdb
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
#import manifold

# Model saved with Keras model.save()
from keras.models import Sequential,Model
from keras.layers.convolutional import Conv2D
import keras.layers
from keras.layers import BatchNormalization,MaxPool2D,Flatten,Dense,Dropout,LSTM
import pickle

#sys.exit()

#scp -i "awskey.pem" -r pdata
#scp -i "awskey.pem" traintime.py
#ec2-user@ec2-3-17-14-185.us-east-2.compute.amazonaws.com


#Randomly split and store data

datalist = ["118","200","207","209","214","217"]
datanum = "100"
xs0 = np.load('dataparse/' + datanum + 'x.npy')
xs = np.load('dataparse/2.' + datanum + 'x.npy')
ys = np.load('dataparse/2.' + datanum + 'y.npy')
for datanum in datalist:
    nx0 = np.load('dataparse/' + datanum + 'x.npy')
    nx = np.load('dataparse/2.' + datanum + 'x.npy')
    ny = np.load('dataparse/2.' + datanum + 'y.npy')
    print(len(xs))
    print(len(xs0))
    xs0 = np.concatenate((xs0,nx0))
    xs = np.concatenate((xs,nx))
    ys = np.concatenate((ys,ny))

indices = np.arange(len(xs))
rind = np.random.permutation(indices)

rxs = xs[rind]
rys = ys[rind]

for i in range(len(rxs)):
    rxs[i] = np.pad(rxs[i],(0,8668-len(rxs[i])),'constant' )
    rxs = np.array(rxs.tolist())

#16225 data points
testxs = rxs[0:3245]
testys = rys[0:3245]
trainxs = rxs[3245:16225]
trainys = rys[3245:16225]
np.save('finaldata/2.testx',testxs)
np.save('finaldata/2.testy',testys)
np.save('finaldata/2.trainx',trainxs)
np.save('finaldata/2.trainy',trainys)

rxs0 = xs0[rind]
trainxs0 = rxs0[3245:16225]
testxs0 = rxs0[0:3245]
np.save('finaldata/trainx',trainxs0)
np.save('finaldata/testx',testxs0)




