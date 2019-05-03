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

    def plot_embedding(X, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i,0], X[i,1], str(y[i]), color=plt.cm.Set1(y[i]/10.),fontdict={'weight':'bold','size':9})

        if hasattr(offsetbox, 'AnnotationBbox'):
            shown_images = np.array([[1., 1.]])
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox - offsetbbox.AnnotationBbox(offsetbox.OffsetImage(digits.images[i],cmap=plt.cm.gray_r),X[i])
                ax.add_artist(imagebox)
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    def Isomap(self):
        X_iso = manifold.Isomap(n_neighbors, n_components=256).fit_transform(self.X)
        return X_iso

    def LLE(self):
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=256, method='standard')
        X_lle = clf.fit_transform(self.X)
        return X_lle

    def LTSA(self):
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=256, method='ltsa')
        X_ltsa = clf.fit_transform(X)
        return X_ltsa

    def Spectral(self):
        embedder = manifold.SpectralEmbedding(n_components=256, random_state=0, eigen_solver="arpack")
        X_se = embedder.fit_transform(self.X)
        return X_se
