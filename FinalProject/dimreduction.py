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
from manifold import dimReduct
import time

xs = np.load('finaldata/trainx.npy')
xtest = np.load('finaldata/testx.npy')

xs2 = np.load('finaldata/2.trainx.npy')
xtest2 = np.load('finaldata/2.testx.npy')
xs2 = xs2.reshape(-1,1,8668)
xtest2 = xtest2.reshape(-1,1,8668)

ys = np.load('finaldata/2.trainy.npy')
ytest = np.load('finaldata/2.testy.npy')


dimtype = sys.argv[1]
if dimtype == "ISOMAP":
    t1 = time.time()
    myman = dimReduct(xs[0:500])
    dxs = myman.Isomap()
    t2 = time.time()
    print("Time will take roughly 800x longer than:")
    print(t2-t1)

    print("start d1")
    myman = dimReduct(xs[0:10000])
    dxs = myman.Isomap()
    xs = dxs.reshape(-1,16,16,1)
    xtest = xs[0:2000]
    xs = xs[2000:10000]

    print("start d2")
    myman2 = dimReduct(xs2[0:10000])
    dxs2 = myman2.Isomap()
    xs2 = dxs2.reshape(-1,1,256)
    xtest2 = xs2[0:2000]
    xs2 = xs2[2000:10000]  

elif dimtype == "LLE":
    t1 = time.time()
    myman = dimReduct(xs[0:500])
    dxs = myman.LLE()
    t2 = time.time()
    print("Time will take roughly 800x longer than:")
    print(t2-t1)

    print("start d1")
    myman = dimReduct(xs[0:10000])
    dxs = myman.LLE()
    xs = dxs.reshape(-1,16,16,1)
    xtest = xs[0:2000]
    xs = xs[2000:10000]

    print("start d2")
    myman2 = dimReduct(xs2[0:10000])
    dxs2 = myman2.LLE()
    xs2 = dxs2.reshape(-1,1,256)
    xtest2 = xs2[0:2000]
    xs2 = xs2[2000:10000]

elif dimtype == "Spectral":
    t1 = time.time()
    myman = dimReduct(xs[0:500])
    dxs = myman.Spectral()
    t2 = time.time()
    print("Time will take roughly 800x longer than:")
    print(t2-t1)

    print("start d1")
    myman = dimReduct(xs[0:10000])
    dxs = myman.Spectral()
    xs = dxs.reshape(-1,16,16,1)
    xtest = xs[0:2000]
    xs = xs[2000:10000]

    print("start d2")
    myman2 = dimReduct(xs2[0:10000])
    dxs2 = myman2.Spectral()
    xs2 = dxs2.reshape(-1,1,256)
    xtest2 = xs2[0:2000]
    xs2 = xs2[2000:10000]

else:
    print("ERROR: invalid dimensionality reduction type (ISOMAP, LLE, or Spectral)")

np.save(dimtype+"/d1xs",xs)
np.save(dimtype+"/d1xt",xtest)
np.save(dimtype+"/d2xs",xs2)
np.save(dimtype+"/d2xt",xtest2)

