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
dimtype = sys.argv[1]
if dimtype == "Normal":
    inputShape = [128,128]
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
if dimtype == "Normal":
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

inputShape2 = 256
if dimtype == "Normal":
    inputShape2 = 8668
# create and fit the LSTM network
batch_size = 64
model2 = Sequential()
model2.add(LSTM(512, return_sequences=True, input_shape= (1,inputShape2) )) #(1, check)))
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



if dimtype == "Normal":
    xs = np.load('finaldata/trainx.npy')
    xtest = np.load('finaldata/testx.npy')
    xs2 = np.load('finaldata/2.trainx.npy')
    xtest2 = np.load('finaldata/2.testx.npy')
    xs2 = xs2.reshape(-1,1,8668)
    xtest2 = xtest2.reshape(-1,1,8668)

elif dimtype == "ISOMAP":
    xs = np.load("ISOMAP/d1xs.npy")
    xtest = np.load("ISOMAP/d1xt.npy")
    xs2 = np.load("ISOMAP/d2xs.npy")
    xtest2 = np.load("ISOMAP/d2xt.npy")

elif dimtype == "LLE":
    xs = np.load("LLE/d1xs.npy")
    xtest = np.load("LLE/d1xt.npy")
    xs2 = np.load("LLE/d2xs.npy")
    xtest2 = np.load("LLE/d2xt.npy")

elif dimtype == "Spectral":
    xs = np.load("Spectral/d1xs.npy")
    xtest = np.load("Spectral/d1xt.npy")
    xs2 = np.load("Spectral/d2xs.npy")
    xtest2 = np.load("Spectral/d2xt.npy")

else:
    print("ERROR: invalid dimensionality reduction type (Normal, ISOMAP, LLE, or Spectral)")

ys = np.load('finaldata/2.trainy.npy')
ytest = np.load('finaldata/2.testy.npy')

ytest = ys[0:2000]
ys = ys[2000:10000]

train1 = True
train2 = True
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

    if dimtype == "Normal":
        np.save("finaldata/final3h",out)
    else:
        np.save(dimtype+"/final3h",out)

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



