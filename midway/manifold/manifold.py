#Implement manifold learning functions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

class manifold(object):
    def __init__(self, X):
        self.X = X

    n_samples, n_features = self.X.shape
    n_neighbors = 30

    def Isomap():
        X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(self.X)

    def LLE():
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
        X_lle = clf.fit_transform(self.X)
