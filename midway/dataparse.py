from __future__ import division, print_function
import json
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import numpy as np
import biosppy
import matplotlib.pyplot as plt
import wfdb

indices = []


#100
#118

#200
#207

#209
#214
#217

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
    if ((ann.sample[acount] - peaks[count]) < -75): # increase acount
        print("increase")
        acount += 1
    if ((ann.sample[acount] - peaks[count]) > 75): # decrease acount
        print("decrease")
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

np.save("dataparse/"+datanum+"x",xs)
np.save("dataparse/"+datanum+"y",labels)


