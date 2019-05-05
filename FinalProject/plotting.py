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
from manifold import dimReduct

h = np.load("finaldata/final3h.npy")
h1 = np.load("ISOMAP/final3h.npy")
h2 = np.load("LLE/final3h.npy")
h3 = np.load("Spectral/final3h.npy")
print("Normal training accuracy")
plt.plot(h[3])
plt.show()
print("Normal validation accuracy")
plt.plot(h[1])
plt.show()
print("Isomap training accuracy")
plt.plot(h1[3])
plt.show()
print("Isomap validation accuracy")
plt.plot(h1[1])
plt.show()
print("LLE training accuracy")
plt.plot(h2[3])
plt.show()
print("LLE validation accuracy")
plt.plot(h2[1])
plt.show()
print("Spectral training accuracy")
plt.plot(h3[3])
plt.show()
print("Spectral validation accuracy")
plt.plot(h3[1])
plt.show()


xs = np.load('finaldata/trainx.npy')
xso = xs[2000:10000]
xs = np.load("Isomap/d1xs.npy")

#Plotting example Isomap transformations
#indices: 0 1 2 3 4 8 12
#labels: 2 1 3 7 4 5 0
#labels: L N / X V R A
plt.gray()
for i in [0,1,2,3,4,8,12]:
    plt.imshow(xso[i,:,:,0])
    plt.show()
    plt.imshow(xs[i,:,:,0])
    plt.show()



