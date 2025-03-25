import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster._kmeans
import scipy.spatial.distance
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

NUM_CLASSES = 10

X,y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)


# Standard MNIST split: 60k training, 10k testing
X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000].astype(int)
y_test = y[60000:].astype(int)


def cluster(X_train, y_train):
    clusters = np.zeros((NUM_CLASSES, 64, 784))
    y_temp = np.zeros(NUM_CLASSES*64)

    for i in range(NUM_CLASSES): 
        Xi_train = X_train[y_train == i]
        kmeans = sklearn.cluster._kmeans.KMeans(n_clusters=64, random_state=9)
        kmeans.fit(Xi_train)
        Ci = kmeans.cluster_centers_
        
        clusters[i] = Ci
        y_temp[i*64:(i+1)*64] = i
        
    return clusters, y_temp

def kNN(X_train, y_train, X_test, k):
    distances = scipy.spatial.distance.cdist(X_test, X_train, 'euclidean')  
    y_pred = np.zeros(len(X_test), dtype=int)

    for i in range(len(X_test)):
        kNN_indices = np.argsort(distances[i])[:k]
        kNN_labels = y_train[kNN_indices].astype(int)
        y_pred[i] = np.argmax(np.bincount(kNN_labels).astype(int))
        
    return y_pred




# TASK 2:
clusters, y_temp = cluster(X_train, y_train)
X_temp = np.reshape(clusters, (NUM_CLASSES*64, 784))

y_pred = kNN(X_temp, y_temp, X_test, 1)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


