from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data for the whole iris dataset
iris_data = np.loadtxt("iris_detection/classes/iris_data", delimiter=",", usecols=(0, 1, 2, 3))
setosa = iris_data[0:50]
versicolor = iris_data[50:100]
virginica = iris_data[100:150]

n_training_samples = 30 # Number of training samples per class

# Functions
def get_training_data(start):
    setosa_train = setosa[start:(start + n_training_samples)]
    versicolor_train = versicolor[start:(start + n_training_samples)]
    virginica_train = virginica[start:(start + n_training_samples)]
    training_data = np.vstack((setosa_train, versicolor_train, virginica_train))
    
    # Training labels
    setosa_labels = np.kron(np.array([1, 0, 0]), np.ones((n_training_samples, 1)))
    versicolor_labels = np.kron(np.array([0, 1, 0]), np.ones((n_training_samples, 1)))
    virginica_labels = np.kron(np.array([0, 0, 1]), np.ones((n_training_samples, 1)))
    training_labels = np.vstack((setosa_labels, versicolor_labels, virginica_labels))
    
    return training_data, training_labels

def get_test_data(start):
    setosa_test = setosa[start:(start + (50-n_training_samples))]
    versicolor_test = versicolor[start:(start + (50-n_training_samples))]
    virginica_test = virginica[start:(start + (50-n_training_samples))]
    test_data = np.vstack((setosa_test, versicolor_test, virginica_test))
    
    # Testing labels
    setosa_labels_test = np.kron(np.array([1, 0, 0]), np.ones((50-n_training_samples, 1)))
    versicolor_labels_test = np.kron(np.array([0, 1, 0]), np.ones((50-n_training_samples, 1)))
    virginica_labels_test = np.kron(np.array([0, 0, 1]), np.ones((50-n_training_samples, 1)))
    test_labels = np.vstack((setosa_labels_test, versicolor_labels_test, virginica_labels_test))
    
    return test_data, test_labels

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_linear_classifier(X, t, alpha, iterations):
    N, D = X.shape
    C = t.shape[1]
    
    # Add bias term
    X_aug = np.hstack((X, np.ones((N, 1))))
    
    # Usikker på hvordan vi skal initialisere vektene
    # Zero initialization
    #W = np.zeros((D + 1, C))  # Shape: (D+1, C)
    
    # Randomly initialize weights
    #W = np.random.uniform(-1, 1, (D + 1, C))
    
    # Initialize weights with Xavier initialization
    #limit = np.sqrt(1 / D)
    #W = np.random.uniform(-limit, limit, size=(D + 1, C))  # Shape: (D+1, C)
    
    # Initialize weights with a small random value
    W = np.random.randn(D+1, C) * 0.3
    
    # Store MSE for each iteration
    losses = []
    
    for _ in range(iterations):
        z = np.dot(X_aug, W)
        g = sigmoid(z)
        
        # Compute the gradient, ikke helt fornøyd med navnene
        error = g - t
        sigmoid_derivative = g * (1 - g)
        weight_gradient = np.dot(X_aug.T, (error * sigmoid_derivative)) / N
        
        # Update weights
        W -= alpha * weight_gradient
        
        # Compute the loss (mean squared error)
        loss = np.mean((g - t) ** 2)
        losses.append(loss)
        
    return W, losses
    
def predict(X, W):
    X_aug = np.hstack((X, np.ones((X.shape[0], 1))))
    
    z = np.dot(X_aug, W)
    g = sigmoid(z)
    
    return g

def confusion_matrix(y_true, y_pred):
    cm = np.zeros((3, 3), dtype=int)
    
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
        
    return cm

### Task 1 ###
alphas = [0.001, 0.01, 0.1, 1]
iterations = 10000
plt.figure(figsize=(10, 6))

# Training data
training_data, training_labels = get_training_data(0)
test_data, test_labels = get_test_data(30)

for alpha in alphas:
    print(f"Training with alpha = {alpha}")
    
    # Train the linear classifier
    weight_matrix, losses = train_linear_classifier(training_data, training_labels, alpha, iterations)
    
    # Predict on the test data
    predictions = predict(test_data, weight_matrix)
    
    # Convert labels to class labels
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Calculate error rate
    error_rate = 1 - accuracy
    print(f"Error rate: {error_rate * 100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)
    
    # Plotting the loss function
    plt.plot(range(iterations), losses, label=f'alpha={alpha}')
    
    print ("-" * 50)

# Plotting the loss function
plt.title("Loss Function vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid()
plt.show()

# Using the last 30 samples of each class for training
alpha = 0.01
iterations = 10000
training_data2, training_labels2 = get_training_data(20)
test_data2, test_labels2 = get_test_data(0)

weight_matrix2, losses2 = train_linear_classifier(training_data2, training_labels2, alpha, iterations)
predictions2 = predict(test_data2, weight_matrix2)
predicted_labels2 = np.argmax(predictions2, axis=1)
true_labels2 = np.argmax(test_labels2, axis=1)

# Calculate accuracy
accuracy2 = np.mean(predicted_labels2 == true_labels2)
print(f"Accuracy with new training set: {accuracy2 * 100:.2f}%")

# Calculate error rate
error_rate2 = 1 - accuracy2
print(f"Error rate with new training set: {error_rate2 * 100:.2f}%")

# Confusion matrix for the new training set
cm2 = confusion_matrix(true_labels2, predicted_labels2)
print("Confusion Matrix with new training set:")
print(cm2)

# Ser ut som det alltid er like bra eller bedre å bruke de første 30 prøvene av hver klasse. Endret på koden, og nå er det motsatt tilfelle?
# Kan prøve litt forskjellige metoder for å finne alpha (for rapportens skyld). Vanligvis tester man seg bare frem
# Vis hva man får dersom man bruker andre algoritmer (vegard brukte innebygde libraries i ML, kan gjøre det her også)

### Task 2 ###
## Plotting histograms for each feature and class
def plot_histogram(feature_index, feature_name):
    plt.figure(figsize=(10, 6))
    plt.hist(setosa[:, feature_index], bins=20, color="red", alpha=0.5, label='Setosa')
    plt.hist(versicolor[:, feature_index], bins=20, color="green", alpha=0.5, label='Versicolor')
    plt.hist(virginica[:, feature_index], bins=20, color="blue", alpha=0.5, label='Virginica')
    plt.legend(loc='upper right')
    plt.title(f"{feature_name} in cm")
    plt.xlabel(feature_name)
    plt.ylabel("Count")
    plt.show()

def plot_all_histograms():
    plot_histogram(0, "Sepal Length (cm)")
    plot_histogram(1, "Sepal Width (cm)")
    plot_histogram(2, "Petal Length (cm)")
    plot_histogram(3, "Petal Width (cm)")

plot_all_histograms()

# Remove the most overlapping feature
iris_data = np.loadtxt("iris_detection/classes/iris_data", delimiter=",", usecols=(0, 2, 3))
setosa = iris_data[0:50]
versicolor = iris_data[50:100]
virginica = iris_data[100:150]

training_data, training_labels = get_training_data(0)
test_data, test_labels = get_test_data(30)

# Train the linear classifier
weight_matrix, losses = train_linear_classifier(training_data, training_labels, alpha, iterations)
predictions = predict(test_data, weight_matrix)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Calculate accuracy and error rate
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy with new feature set: {accuracy * 100:.2f}%")
error_rate = 1 - accuracy
print(f"Error rate with new feature set: {error_rate * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix with new feature set:")
print(cm)

# Remove the 2 most overlapping features
iris_data = np.loadtxt("iris_detection/classes/iris_data", delimiter=",", usecols=(2, 3))
setosa = iris_data[0:50]
versicolor = iris_data[50:100]
virginica = iris_data[100:150]

training_data, training_labels = get_training_data(0)
test_data, test_labels = get_test_data(30)

# Train the linear classifier
weight_matrix, losses = train_linear_classifier(training_data, training_labels, alpha, iterations)
predictions = predict(test_data, weight_matrix)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Calculate accuracy and error rate
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy with only two features: {accuracy * 100:.2f}%")
error_rate = 1 - accuracy
print(f"Error rate with only two features: {error_rate * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix with only two features:")
print(cm)

# Remove the 3 most overlapping features
iris_data = np.loadtxt("iris_detection/classes/iris_data", delimiter=",", usecols=(3))
iris_data = np.expand_dims(iris_data, axis=1) # Needs to be 2D for the classifier

setosa = iris_data[0:50]
versicolor = iris_data[50:100]
virginica = iris_data[100:150]

training_data, training_labels = get_training_data(0)
test_data, test_labels = get_test_data(30)

# Train the linear classifier
weight_matrix, losses = train_linear_classifier(training_data, training_labels, alpha, iterations)
predictions = predict(test_data, weight_matrix)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Calculate accuracy and error rate
accuracy = np.mean(predicted_labels == true_labels)
print(f"Accuracy with only one feature: {accuracy * 100:.2f}%")
error_rate = 1 - accuracy
print(f"Error rate with only one feature: {error_rate * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix with only one feature:")
print(cm)