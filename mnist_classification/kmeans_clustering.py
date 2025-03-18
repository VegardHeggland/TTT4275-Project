import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import sklearn.cluster._kmeans

X,y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)


kmeans = sklearn.cluster._kmeans.KMeans(n_clusters=10, random_state=9)
id_xi = kmeans.fit_predict(X)
Ci = kmeans.cluster_centers_


#print(id_xi)