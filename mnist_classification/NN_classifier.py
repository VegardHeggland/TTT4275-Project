import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import scipy.spatial.distance

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(X[:1000], y[:1000], test_size=0.2, random_state=9)

k = 1

distances = scipy.spatial.distance.cdist(X_test, X_train, 'euclidean')

y_pred = [0 for element in range(len(X_test))]

for i in range(len(X_test)):
    k_nearest_indices = np.argsort(distances[i])[:k]
    k_nearest_labels = y_train[k_nearest_indices].astype(int)
    y_pred[i] = np.argmax(np.bincount(k_nearest_labels).astype(int))
    
# TODO: Remember indices of pictures difficuilt to classify and plot them


