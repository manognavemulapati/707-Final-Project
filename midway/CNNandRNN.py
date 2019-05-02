from __future__ import division, print_function
import json
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import pandas as pd
import numpy as np
import biosppy
import matplotlib.pyplot as plt
import wfdb
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import manifold

# # Flask utils
# from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# Define a flask app
#app = Flask(__name__)


# Model saved with Keras model.save()
from keras.models import Sequential,Model
from keras.layers.convolutional import Conv2D
import keras.layers
from keras.layers import BatchNormalization,MaxPool2D,Flatten,Dense,Dropout,LSTM

IMAGE_SIZE = [128,128]
inputShape = [64,64]

number_of_classes = 8

model = Sequential()
model.add(Conv2D(64, (3,3),strides = (1,1), input_shape = inputShape + [1],kernel_initializer='glorot_uniform'))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
model.add(Flatten())
model.add(Dense(2048))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# create and fit the LSTM network
batch_size = 64
model2 = Sequential()
model2.add(LSTM(512, return_sequences=True, input_shape= inputShape )) #(1, check)))
#model.add(Dropout(0.25))
model2.add(LSTM(256, return_sequences=True))
#model.add(Dropout(0.25))
model2.add(LSTM(128, return_sequences=True))
#model.add(Dropout(0.25))
model2.add(LSTM(64, return_sequences=True))
#model.add(Dropout(0.25))
model2.add(LSTM(32))
model2.add(Dense(2048))
model2.add(Dense(number_of_classes, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Load your trained model
#model = load_model('models/ecgScratchEpoch2.hdf5')
#model._make_predict_function()          # Necessary
#print('Model loaded. Start serving...')

# h = model.fit()
# print(h.history["accuracy"])
output = []
# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')





#def model_predict(file, model):
# if (True):
flag = 1
path = "uploads/foo.csv"
#index1 = str(path).find('sig-2') + 6
#index2 = -4
#ts = int(str(path)[index1:index2])
APC, NORMAL, LBB, PVC, PAB, RBB, VEB = [], [], [], [], [], [], []
output.append(str(path))
result = {"APC": APC, "Normal": NORMAL, "LBB": LBB, "PAB": PAB, "PVC": PVC, "RBB": RBB, "VEB": VEB}


indices = []

#360 points per second, 5 minutes = 108000
#30:06 = 650160
# but max value is 650000
sampto = 650000
datanum = "217"

kernel = np.ones((4,4),np.uint8)
record = wfdb.rdsamp('data/'+datanum, sampto=sampto)
data = record[0][:,0]

signals = []
count = 1
peaks =  biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate = 200)[0]
#print(len(peaks))
for i in (peaks[1:-1]):
    diff1 = abs(peaks[count - 1] - i)
    diff2 = abs(peaks[count + 1]- i)
    x = peaks[count - 1] + diff1//2
    y = peaks[count + 1] - diff2//2
    signal = data[x:y]
    signals.append(signal)
    count += 1
    indices.append((x,y))

datlength = len(signals)
labels = []
xs = []

acount = 1
for count, i in enumerate(signals):
    if (count%10 == 0):
      print(count)
    fig = plt.figure(frameon=False)
    plt.plot(i) 
    plt.xticks([]), plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    filename = 'fig' + '.png'
    fig.savefig(filename)
    im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    im_gray = cv2.erode(im_gray,kernel,iterations = 1)
    im_gray = cv2.resize(im_gray, (128, 128), interpolation = cv2.INTER_LANCZOS4)
    im_gray = np.expand_dims(im_gray,axis=2)
    plt.close(fig)

    ann = wfdb.rdann("data/"+datanum,'atr',sampto=sampto)

    # fail safe in case christov_segmenter misses a peak
    if ((ann.sample[acount] - peaks[count]) < 75): # increase acount
        acount += 1
    if ((ann.sample[acount] - peaks[count]) > 75): # decrease acount
        acount -= 1

    label = ann.symbol[acount]
    y = np.array(np.zeros(8, dtype=np.float32))[None,:]
    if (label == "A"):
       y[0][0] = 1.0
    elif (label == "N"):
       y[0][1] = 1.0
    elif (label == "L"):
       y[0][2] = 1.0
    elif (label == "/"):
       y[0][3] = 1.0
    elif (label == "V"):
       y[0][4] = 1.0
    elif (label == "R"):
       y[0][5] = 1.0
    elif (label == "E"):
       y[0][6] = 1.0
    else:
       y[0][7] = 1.0
    labels.append(y[0])
    xs.append((im_gray))
    acount += 1

labels = np.array(labels)
xs = np.array(xs)
#xs2 = xs.reshape(-1,128,128)

dimReduction = manifold.manifold(xs)
xs1 = dimReduction.PCA()
inputShape = xs1.shape
xs1 = xs1.reshape(xs.shape[0],64,64,1)
xs2 = xs1.reshape(-1,64,64)

hist = model.fit(xs,labels,epochs=100,batch_size=64)
hist2 = model2.fit(xs2,labels,epochs=100,batch_size=64)

#27 layers
intmodel1 = Model(inputs=model.input,outputs=model.layers[25].output)
#7 layers
intmodel2 = Model(inputs=model2.input,outputs=model2.layers[5].output)
#print(hist.history.keys())

print("predict with 1")
int1 = intmodel1.predict(xs)
print("predict with 2")
int2 = intmodel2.predict(xs2)

intxs = int1 * int2

model3 = Sequential()
model3.add(Dense(number_of_classes, activation='softmax',input_shape=[2048]))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist3 = model3.fit(intxs,labels,epochs=100,batch_size=64)



# plt.plot(hist.history['acc'])
# plt.show()
#plt.plot(hist.history['loss'])


#intmodel = Model(inputs=model.input,outputs=model.layers[25].output)


#     pred = model.predict(im_gray.reshape((1, 128, 128, 3)))
#     pred_class = pred.argmax(axis=-1)
#     if pred_class == 0:
#         APC.append(indices[count]) 
#     elif pred_class == 1:
#         NORMAL.append(indices[count]) 
#     elif pred_class == 2:    
#         LBB.append(indices[count])
#     elif pred_class == 3:
#         PAB.append(indices[count])
#     elif pred_class == 4:
#         PVC.append(indices[count])
#     elif pred_class == 5:
#         RBB.append(indices[count]) 
#     elif pred_class == 6:
#         VEB.append(indices[count])

# result = sorted(result.items(), key = lambda y: len(y[1]))[::-1]   
# output.append(result)


# data = {}
# data['filename'+ str(flag)] = str(path)
# data['result'+str(flag)] = str(result)

# json_filename = 'data.txt'
# with open(json_filename, 'a+') as outfile:
#     json.dump(data, outfile) 
# flag+=1 

# >>> np.save('pdata/'+datanum+'x',xs)
# >>> np.save('pdata/'+datanum+'y',labels)
# >>> asdf = np.load('pdata/100x.npy')


# with open(json_filename, 'r') as file:
#     filedata = file.read()
# filedata = filedata.replace('}{', ',')
# with open(json_filename, 'w') as file:
#     file.write(filedata) 
os.remove('fig.png')      
print(output)

#100
#118

#200
#207

#209
#214
#217



