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
for datanum in ['100','118','200','207','209','214','217']:
  sampto = 650000

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

      im_gray = i

      ann = wfdb.rdann("data/"+datanum,'atr',sampto=sampto)

      # fail safe in case christov_segmenter misses a peak
      while ((ann.sample[acount] - peaks[count]) < -75): # increase acount
          print("increase")
          acount += 1
      while ((ann.sample[acount] - peaks[count]) > 75): # decrease acount
          print("decrease")
          acount -= 1

      label = ann.symbol[acount]
      y = np.array(np.zeros(8, dtype=np.float32))[None,:]
      if (label == "A"):    #Atrial premature beat
         y[0][0] = 1.0
      elif (label == "N"):  #Normal beat
         y[0][1] = 1.0
      elif (label == "L"):  #Left bundle branch block beat
         y[0][2] = 1.0
      elif (label == "/"):  #Paced beat
         y[0][3] = 1.0
      elif (label == "V"):  #Premature ventricular contraction
         y[0][4] = 1.0
      elif (label == "R"):  #Right bundle branch block beat
         y[0][5] = 1.0      
      elif (label == "E"):  #Ventricular escape beat
         y[0][6] = 1.0      
      else:                 #Unknown
         y[0][7] = 1.0      
      labels.append(y[0])
      xs.append((im_gray))
      acount += 1

  labels = np.array(labels)
  xs = np.array(xs)

  np.save("dataparse/2."+datanum+"x",xs)
  np.save("dataparse/2."+datanum+"y",labels)


