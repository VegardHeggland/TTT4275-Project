import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster._kmeans
import scipy.spatial.distance
import time
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

NUM_CLASSES = 10

X,y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Standard MNIST split: 60k training, 10k testing
X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000].astype(int)
y_test = y[60000:].astype(int)

def kNN(X_train, y_train, X_test, k):
    distances = scipy.spatial.distance.cdist(X_test, X_train, 'euclidean')  
    y_pred = np.zeros(len(X_test), dtype=int)

    for i in range(len(X_test)):
        kNN_indices = np.argsort(distances[i])[:k]
        kNN_labels = y_train[kNN_indices].astype(int)
        y_pred[i] = np.argmax(np.bincount(kNN_labels).astype(int))
        
    return y_pred

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

# TASK 1 - Nearest Neighbor:
def runTask1(X_train, y_train, X_test, y_test):
    time_start = time.time()
    y_pred = kNN(X_train, y_train, X_test, 1)
    time_end = time.time()

    error_rate = np.mean(y_pred != y_test)
    
    print(f"Nearest Neighbor:")
    print(f"Error rate: {error_rate}")
    print(f"Execution time: {time_end - time_start} seconds")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Nearest Neighbor", fontsize=14, fontweight='bold')
    plt.savefig("mnist_classification/figures/confusion_matrix_k1.eps", format='eps')
    plt.show()

    # Missclassified images
    misclassified_indices = np.where(y_pred != y_test)[0]
    fig, ax = plt.subplots(1, 4, figsize=(10, 4))
    for i in range(10):
        ax[i].set_title(f"Predicted: {y_pred[misclassified_indices[i]]}\nTrue: {y_test[misclassified_indices[i]]}", fontsize=12)
        ax[i].imshow(X_test[misclassified_indices[i]].reshape(28, 28))
        ax[i].axis('off')

    plt.suptitle("Misclassified Images", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("mnist_classification/figures/misclassified.eps", format='eps')
    plt.show()

    # Correctly classified images
    correctly_classified_indices = np.where(y_pred == y_test)[0]
    fig, ax = plt.subplots(1, 4, figsize=(10, 4))
    for i in range(4):
        ax[i].set_title(f"Predicted: {y_pred[correctly_classified_indices[i]]}\nTrue: {y_test[correctly_classified_indices[i]]}", fontsize=12)
        ax[i].imshow(X_test[correctly_classified_indices[i]].reshape(28, 28))
        ax[i].axis('off')
    
    plt.suptitle("Correctly Classified Images", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("mnist_classification/figures/correctly_classified.eps", format='eps')
    plt.show()

# TASK 2 - k-Nearest Neigbors with clustering:
def runTask2(X_train, y_train, X_test, y_test):
   
    # k-NN with clustering and k=1:
    time_start = time.time()
    clusters, y_temp = cluster(X_train, y_train)
    X_temp = np.reshape(clusters, (NUM_CLASSES*64, 784))

    y_pred_1 = kNN(X_temp, y_temp, X_test, 1)
    time_end = time.time()
    error_rate = np.mean(y_pred_1 != y_test)

    print(f"k-NN with clustering and k=1")
    print(f"Error rate: {error_rate}")
    print(f"Execution time: {time_end - time_start} seconds")
    
    # k-NN with clustering and k=7:
    time_start = time.time()
    clusters, y_temp = cluster(X_train, y_train)
    X_temp = np.reshape(clusters, (NUM_CLASSES*64, 784))

    y_pred_7 = kNN(X_temp, y_temp, X_test, 7)
    time_end = time.time()
    error_rate = np.mean(y_pred_7 != y_test)

    print(f"k-NN with clustering and k=7")
    print(f"Error rate: {error_rate}")
    print(f"Execution time: {time_end - time_start} seconds")

    # Subplot comparing the two confusion matrices
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    cm1 = confusion_matrix(y_test, y_pred_1)
    cm2 = confusion_matrix(y_test, y_pred_7)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
    disp1.plot(cmap=plt.cm.Blues, ax=ax[0])
    disp2.plot(cmap=plt.cm.Blues, ax=ax[1])
    ax[0].set_title("k=1", fontsize = 14, fontweight='bold')
    ax[1].set_title("k=7", fontsize = 14, fontweight='bold')
    plt.suptitle("Confusion Matrices for k-NN with Clustering", fontsize=18, fontweight='bold')
    plt.savefig("confusion_matrix_k1_k7_clustered.eps", format='eps')
    plt.show()


runTask1(X_train, y_train, X_test, y_test)
runTask2(X_train, y_train, X_test, y_test)

