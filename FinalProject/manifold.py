#Implement manifold learning functions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

class dimReduct(object):
    def __init__(self, X):
        self.X = X.reshape(X.shape[0], -1)
        self.n_samples, self.n_features = self.X.shape
        self.n_neighbors = 30

    def Isomap(self):
        X_iso = manifold.Isomap(n_neighbors=self.n_neighbors, n_components=256).fit_transform(self.X)
        return X_iso

    def LLE(self):
        clf = manifold.LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=256, method='standard')
        X_lle = clf.fit_transform(self.X)
        return X_lle

    def LTSA(self):
        clf = manifold.LocallyLinearEmbedding(n_neighbors=self.n_neighbors, n_components=256, method='ltsa')
        X_ltsa = clf.fit_transform(X)
        return X_ltsa

    def Spectral(self):
        embedder = manifold.SpectralEmbedding(n_components=256, random_state=0, eigen_solver="arpack")
        X_se = embedder.fit_transform(self.X)
        return X_se