import torch
import numpy as np
import sklearn
from sklearn.decomposition import PCA


x = np.random.randn(10, 15)
pca = PCA(n_components=3)
pca.fit(x)

print(pca.components_.T)
