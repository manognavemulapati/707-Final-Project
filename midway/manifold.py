#Implement manifold learning functions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

class manifold(object):
    def __init__(self, X):
        self.X = X.reshape(X.shape[0], -1)
        self.n_samples, self.n_features = self.X.shape
        self.n_neighbors = 30

    #Try PCA to see if linear dimensionality reduction works
    def PCA(self):
        X_pca = decomposition.TruncatedSVD(n_components=4096).fit_transform(self.X)
        return X_pca

    def Isomap(self):
        X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(self.X)

    def LLE(self):
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
        X_lle = clf.fit_transform(self.X)
