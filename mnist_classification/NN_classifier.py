import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import scipy.spatial.distance
import time



X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

k = 1
chunk_size = 1000

successful_predictions = []
failed_predictions = []

time_start = time.time()
for i in range(0, len(X), chunk_size):
    X_train, X_test, y_train, y_test = train_test_split(X[i:i+chunk_size], y[i:i+chunk_size], test_size=0.2, random_state=9)
    
    distances = scipy.spatial.distance.cdist(X_test, X_train, 'euclidean')
    print(distances.shape)
    for j in range(len(X_test)):
        kNN_indices = np.argsort(distances[j])[:k]
        kNN_labels = y_train[kNN_indices].astype(int)

        y_pred = np.argmax(np.bincount(kNN_labels).astype(int))
        
        if y_pred == int(y_test[j]):

            successful_predictions.append([y_pred, y_test[j], X_test[j]])
        else:
            failed_predictions.append([y_pred, y_test[j], X_test[j]])
time_end = time.time()
print(f'Total time: {time_end - time_start}')
print(f"Error rate: {1 - len(successful_predictions) / (len(successful_predictions) + len(failed_predictions))}")
print(f"Pred: {failed_predictions[2][0]}")
print(f"Actual: {failed_predictions[2][1]}")
plt.imshow(failed_predictions[2][2].reshape(28, 28), cmap='gray')
plt.show()

# k = 1
# chunk_size = 100
# successful_predictions = []
# failed_predictions = []

# time_start = time.time()
# iter = 0

# for i in range(0, len(X_test), chunk_size):
#     time_iter_start = time.time()
#     success_count = 0

#     X_test_chunk = X_test[i:i+chunk_size]
#     distances = scipy.spatial.distance.cdist(X_test_chunk, X_train, 'euclidean')
   
#     y_pred_chunk = np.zeros(chunk_size)
    
#     for j in range(chunk_size):
#         kNN_indices = np.argsort(distances[j])[:k]
#         kNN_labels = y_train[kNN_indices].astype(int)
        
#         y_pred_chunk[j] = np.argmax(np.bincount(kNN_labels).astype(int))
        
#         if y_pred_chunk[j] == y_test[i+j]:
#             successful_predictions.append([y_pred_chunk[j], y_test[i+j], X_test_chunk[j]])
#             success_count += 1
#         else:
#             failed_predictions.append([y_pred_chunk[j], y_test[i+j], X_test_chunk[j]])

#     accuracy = success_count / chunk_size
#     iter += 1
#     time_iter_end = time.time()
#     print(f"| {iter} | Iteration Accuracy: {accuracy} | Iteration time: {time_iter_end - time_iter_start} |")


# time_end = time.time()
# print(f'Total time: {time_end - time_start}')



# distances = scipy.spatial.distance.cdist(X_test, X_train, 'euclidean')

# y_pred = [0 for element in range(len(X_test))]

# for i in range(len(X_test)):
#     k_nearest_indices = np.argsort(distances[i])[:k]
#     k_nearest_labels = y_train[k_nearest_indices].astype(int)
#     y_pred[i] = np.argmax(np.bincount(k_nearest_labels).astype(int))
    
# # TODO: Remember indices of pictures difficuilt to classify and plot them


