import numpy as np
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)

n_samples, n_features = X.shape
n_neighbors = 30

clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
X_lle = clf.fit_transform(X)

