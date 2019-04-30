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

# opencv-python
# pandas
# biosppy
# wfdb
# numpy
# keras

# Model saved with Keras model.save()
# Load your trained model
model = load_model('models/ecgScratchEpoch2.hdf5')
model._make_predict_function()          # Necessary
print('Model loaded.')

if (True):
    APC, NORMAL, LBB, PVC, PAB, RBB, VEB = [], [], [], [], [], [], []
    result = {"APC": APC, "Normal": NORMAL, "LBB": LBB, "PAB": PAB, "PVC": PVC, "RBB": RBB, "VEB": VEB}

    indices = []

    sampto = 1000
    datanum = "100"

    kernel = np.ones((4,4),np.uint8)
    record = wfdb.rdsamp('data/'+datanum, sampto=sampto)
    data = record[0][:,0]
    signals = []
    count = 1
    peaks =  biosppy.signals.ecg.christov_segmenter(signal=data, sampling_rate = 200)[0]
    for i in (peaks[1:-1]):
       diff1 = abs(peaks[count - 1] - i)
       diff2 = abs(peaks[count + 1]- i)
       x = peaks[count - 1] + diff1//2 # selecting a range around the peak
       y = peaks[count + 1] - diff2//2
       signal = data[x:y]
       signals.append(signal)
       count += 1
       indices.append((x,y))

    datlength = len(signals)
    success = 0
    total = 0
    count2 = 0
        
    for count, i in enumerate(signals):
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
        cv2.imwrite(filename, im_gray)
        im_gray = cv2.imread(filename)

        ann = wfdb.rdann("data/100",'atr',sampto=sampto)
        while (ann.sample[count2]-peaks[count] < -50):
            count2 += 1
        label = ann.symbol[count2]
        y = np.array(np.zeros(7, dtype=np.float32))[None,:]
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

        pred = model.predict(im_gray.reshape((1, 128, 128, 3)))
        pred_class = pred.argmax(axis=-1)
        total += 1
        if pred_class == 0:
            if (label == "A"):
                success += 1
            APC.append(indices[count]) 
        elif pred_class == 1:
            if (label == "N"):
                success += 1
            NORMAL.append(indices[count]) 
        elif pred_class == 2:  
            if (label == "L"):
                success += 1  
            LBB.append(indices[count])
        elif pred_class == 3:
            if (label == "/"):
                success += 1
            PAB.append(indices[count])
        elif pred_class == 4:
            if (label == "V"):
                success += 1
            PVC.append(indices[count])
        elif pred_class == 5:
            if (label == "R"):
                success += 1
            RBB.append(indices[count]) 
        elif pred_class == 6:
            if (label == "E"):
                success += 1
            VEB.append(indices[count])

    result = sorted(result.items(), key = lambda y: len(y[1]))[::-1]   

    file_out = 'output.txt'
    open(file_out, 'w').write(str(result)) 

    os.remove('fig.png')      
    accuracy = success/total
    print("Number of Peaks: " + str(len(peaks)-2))
    print("Total Correct: " + str(success))
    print("Accuracy: " + str(accuracy))




