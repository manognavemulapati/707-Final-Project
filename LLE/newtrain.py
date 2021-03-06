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

#sys.exit()

#scp -i "awskey.pem" -r finaldata ec2-user@ec2-18-191-93-216.us-east-2.compute.amazonaws.com:
#scp -i "awskey.pem" newtrain.py ec2-user@ec2-18-191-93-216.us-east-2.compute.amazonaws.com:
#scp -i "awskey.pem" manifold.py ec2-user@ec2-18-191-93-216.us-east-2.compute.amazonaws.com:
#ec2-18-191-93-216.us-east-2.compute.amazonaws.com

#scp -i "awskey.pem" -r finaldata ec2-user@ec2-3-15-26-49.us-east-2.compute.amazonaws.com:
#scp -i "awskey.pem" newtrain.py ec2-user@ec2-3-15-26-49.us-east-2.compute.amazonaws.com:
#scp -i "awskey.pem" manifold.py ec2-user@ec2-3-15-26-49.us-east-2.compute.amazonaws.com:
#ec2-3-15-26-49.us-east-2.compute.amazonaws.com

inputShape = [16,16]
number_of_classes = 8
'''
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
# model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
# model.add(keras.layers.ELU())
# model.add(BatchNormalization())
# model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))
# model.add(keras.layers.ELU())
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
model.add(Flatten())
model.add(Dense(2048))
model.add(keras.layers.ELU())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
'''

# create and fit the LSTM network
batch_size = 64
model2 = Sequential()
model2.add(LSTM(512, return_sequences=True, input_shape= (1,256) )) #(1, check)))
model2.add(Dropout(0.25))
model2.add(LSTM(256, return_sequences=True))
model2.add(Dropout(0.25))
model2.add(LSTM(128, return_sequences=True))
model2.add(Dropout(0.25))
model2.add(LSTM(64, return_sequences=True))
model2.add(Dropout(0.25))
model2.add(LSTM(32))
model2.add(Dense(number_of_classes, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Load your trained model
#model = load_model('models/ecgScratchEpoch2.hdf5')
#model._make_predict_function()          # Necessary
#print('Model loaded. Start serving...')

# h = model.fit()
# print(h.history["accuracy"])
# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


#Randomly split and store data

# datalist = ["118","200","207","209","214","217"]
# datanum = "100"
# xs = np.load('dataparse/' + datanum + 'x.npy')
# ys = np.load('dataparse/' + datanum + 'y.npy')
# for datanum in datalist:
#     nx = np.load('dataparse/' + datanum + 'x.npy')
#     ny = np.load('dataparse/' + datanum + 'y.npy')
#     print(len(xs))
#     xs = np.concatenate((xs,nx))
#     ys = np.concatenate((ys,ny))

# indices = np.arange(len(xs))
# rind = np.random.permutation(indices)
# rxs = xs[rind]
# rys = ys[rind]
# #16225 data points
# testxs = rxs[0:3245]
# testys = rys[0:3245]
# trainxs = rxs[3245:16225]
# trainys = rys[3245:16225]
# np.save('finaldata/testx',testxs)
# np.save('finaldata/testy',testys)
# np.save('finaldata/trainx',trainxs)
# np.save('finaldata/trainy',trainys)

# >>> np.save('pdata/'+datanum+'x',xs)
# >>> np.save('pdata/'+datanum+'y',labels)

#xs2 = xs.reshape(-1,128,128)

#dimReduction = manifold.manifold(xs)
#xs1 = dimReduction.PCA()
#inputShape = xs1.shape
#xs1 = xs1.reshape(xs.shape[0],64,64,1)
#xs2 = xs1.reshape(-1,64,64)




#pickle_in = open("chck/d0.pickle","rb")
#pin = pickle.load(pickle_in)

xs = np.load('finaldata/trainx.npy')
xtest = np.load('finaldata/testx.npy')

xs2 = np.load('finaldata/2.trainx.npy')
xtest2 = np.load('finaldata/2.testx.npy')
xs2 = xs2.reshape(-1,1,8668)
xtest2 = xtest2.reshape(-1,1,8668)

ys = np.load('finaldata/2.trainy.npy')
ytest = np.load('finaldata/2.testy.npy')
'''
print("hello!")
import time
t1 = time.time()
myman = dimReduct(xs[0:500])
print("asdf")
dxs = myman.Isomap()
t2 = time.time()
print(t2-t1)

print("start d1")

myman = dimReduct(xs[0:10000])
dxs = myman.LLE()
xs = dxs.reshape(-1,16,16,1)
xtest = xs[0:2000]
xs = xs[2000:10000]

# myman = dimReduct(xtest)
# dxs = myman.LLE()
# xtest = dxs.reshape(-1,16,16,1)

print("start d2")
myman2 = dimReduct(xs2[0:10000])
dxs2 = myman2.LLE()
xs2 = dxs2.reshape(-1,1,256)
xtest2 = xs2[0:2000]
xs2 = xs2[2000:10000]

# myman2 = dimReduct(xtest2)
# dxs2 = myman2.LLE()
# xtest2 = dxs2.reshape(-1,1,256)

# sys.exit()
np.save("chck/d1xs",xs)
np.save("chck/d1xt",xtest)
np.save("chck/d2xs",xs2)
np.save("chck/d2xt",xtest2)
'''
xs = np.load("chck/d1xs.npy")
xtest = np.load("chck/d1xt.npy")
xs2 = np.load("chck/d2xs.npy")
xtest2 = np.load("chck/d2xt.npy")

ytest = ys[0:2000]
ys = ys[2000:10000]

train1 = False
train2 = False
train3 = True

if (train1):
    class csaver(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            self.model.save("chck/weights1.hdf5")
            out = np.array([logs['val_loss'],logs['val_acc'],logs['loss'],logs['acc']])
            np.save("chck/1d{}".format(epoch),out.reshape(-1,1))

    checkpointer = csaver()

    #sys.exit()
    vdata = (xtest,ytest)

    hist = model.fit(xs,ys,epochs=100,batch_size=64,validation_data=vdata,callbacks=[checkpointer])
    hist = hist.history

    out = np.array([hist['val_loss'],hist['val_acc'],hist['loss'],hist['acc']])
    np.save("chck/final1h",out)


if (train2):
    class csaver2(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            self.model.save("chck/weights2.hdf5")
            out = np.array([logs['val_loss'],logs['val_acc'],logs['loss'],logs['acc']])
            np.save("chck/2d{}".format(epoch),out.reshape(-1,1))

    checkpointer = csaver2()

    vdata = (xtest2,ytest)
    hist2 = model2.fit(xs2,ys,epochs=100,batch_size=64,validation_data=vdata,callbacks=[checkpointer])
    hist2 = hist2.history

    out = np.array([hist2['val_loss'],hist2['val_acc'],hist2['loss'],hist2['acc']])
    np.save("chck/final2h",out)


if (train3):
    #model0 = load_model('chck/weights.hdf5')
    #model0._make_predict_function()  
    #model.make_predict_function()
    if (train1 == False):
        model = load_model('chck/weights1.hdf5')
    #model2 = load_model('chck/weights2.hdf5')
    if (train2 == False):
        model2 = load_model('chck/weights2.hdf5')
    #27 layers
    intmodel1 = Model(inputs=model.input,outputs=model.layers[18].output)
    #7 layers
    intmodel2 = Model(inputs=model2.input,outputs=model2.layers[8].output)
    #print(hist.history.keys())

    print("predict with 1")
    int1 = intmodel1.predict(xs)
    print("predict with 2")
    int2 = intmodel2.predict(xs2)

    print("predict with 1")
    t1 = intmodel1.predict(xtest)
    print("predict with 2")
    t2 = intmodel2.predict(xtest2)

    #intxs = int1 * int2

    #from keras import backend as K
    #int2 = K.variable(int2)
    #int1 = K.variable(int1)
    '''
    model3 = Sequential()
    model3.add(Dense(2048,input_shape=[32]))
    model3.add(Merge([int1],mode='dot'))
    model3.add(Dense(number_of_classes, activation='softmax',input_shape=[2048]))
    '''
    input1 = keras.layers.Input(shape=(32,))
    x1 = keras.layers.Dense(2048)(input1)

    input2 = keras.layers.Input(shape=(2048,))
    multed = keras.layers.Multiply()([x1, input2])

    output = keras.layers.Dense(8, activation='softmax')(multed)
    model3 = keras.models.Model(inputs=[input1, input2], outputs=output)

    model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("begin fit")
    hist3 = model3.fit([int2,int1],ys,epochs=100,batch_size=64,validation_data = ([t2,t1],ytest))
    #print(hist3.history)
    model3.save("chck/weights3.hdf5")
    hist3 = hist3.history

    out = np.array([hist3['val_loss'],hist3['val_acc'],hist3['loss'],hist3['acc']])
    np.save("chck/final3h",out)

'''
model3 = Sequential()
model3.add(Dense(number_of_classes, activation='softmax',input_shape=[2048]))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist3 = model3.fit(intxs,ys,epochs=100,batch_size=64)
'''


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

#100
#118

#200
#207

#209
#214
#217



