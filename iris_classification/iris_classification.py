import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Constants
n_training_samples = 30
iterations = 3000

### Functions ###
def get_training_data(data, start):
    setosa = data[0:50]
    versicolor = data[50:100]
    virginica = data[100:150]
    
    setosa_train = setosa[start:(start + n_training_samples)]
    versicolor_train = versicolor[start:(start + n_training_samples)]
    virginica_train = virginica[start:(start + n_training_samples)]
    training_data = np.vstack((setosa_train, versicolor_train, virginica_train))
    
    setosa_labels = np.kron(np.array([1, 0, 0]), np.ones((n_training_samples, 1)))
    versicolor_labels = np.kron(np.array([0, 1, 0]), np.ones((n_training_samples, 1)))
    virginica_labels = np.kron(np.array([0, 0, 1]), np.ones((n_training_samples, 1)))
    training_labels = np.vstack((setosa_labels, versicolor_labels, virginica_labels))
    
    return training_data, training_labels

def get_test_data(data, start):
    setosa = data[0:50]
    versicolor = data[50:100]
    virginica = data[100:150]
    
    setosa_test = setosa[start:(start + (50-n_training_samples))]
    versicolor_test = versicolor[start:(start + (50-n_training_samples))]
    virginica_test = virginica[start:(start + (50-n_training_samples))]
    test_data = np.vstack((setosa_test, versicolor_test, virginica_test))
    
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
    X_aug = np.hstack((X, np.ones((N, 1)))).T
    t = t.T
    
    W = np.zeros((C, D+1))
    losses = []
    
    for _ in range(iterations):
        z = np.dot(W, X_aug)
        g = sigmoid(z)
        
        # Compute the gradient
        error = g - t
        sigmoid_derivative = g * (1 - g)
        mse_grad_W = np.dot(error * sigmoid_derivative, X_aug.T)
        
        # Update weights
        W -= alpha * mse_grad_W
        
        # Compute the mean square error
        loss = np.mean((g - t) ** 2)
        losses.append(loss)
        
    return W, losses
    
def predict(X, W):
    X_aug = np.hstack((X, np.ones((X.shape[0], 1)))).T
    z = np.dot(W, X_aug)
    g = sigmoid(z)
    
    return g.T

def make_confusion_matrix(y_true, y_pred):
    confusion_matrix = np.zeros((3, 3), dtype=int)
    
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1
        
    return confusion_matrix

def plot_confusion_matrices(confusion_matrix_training, title_training, confusion_matrix_test, title_test):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    disp_train = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_training, display_labels=['Setosa', 'Versicolor', 'Virginica'])
    disp_train.plot(ax=axs[0], cmap=plt.cm.Blues, colorbar=True)
    axs[0].set_title(title_training)
    axs[0].set_ylabel('True label')
    axs[0].set_xlabel('Predicted label')
    axs[0].set_yticklabels(['Setosa', 'Versicolor', 'Virginica'], rotation=90)

    disp_test = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_test, display_labels=['Setosa', 'Versicolor', 'Virginica'])
    disp_test.plot(ax=axs[1], cmap=plt.cm.Blues, colorbar=True)
    axs[1].set_title(title_test)
    axs[1].set_ylabel('True label')
    axs[1].set_xlabel('Predicted label')
    axs[1].set_yticklabels(['Setosa', 'Versicolor', 'Virginica'], rotation=90)

    plt.show()

def plot_histogram(data, feature_index, feature_name, ax):
    setosa = data[0:50]
    versicolor = data[50:100]
    virginica = data[100:150]

    data_sets = [
        (setosa[:, feature_index], "red", "Setosa"),
        (versicolor[:, feature_index], "green", "Versicolor"),
        (virginica[:, feature_index], "blue", "Virginica")
    ]

    for data, color, label in data_sets:
        ax.hist(data, bins=20, density=True, alpha=0.5, color=color, label=label)
        sns.kdeplot(data, color=color, linewidth=2, alpha = 0.5, ax=ax)

    ax.set_title(f"{feature_name} in cm")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.legend(loc='upper right')

def plot_all_histograms(data):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    
    feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]  
    
    for i in range(4):
        plot_histogram(data, i, feature_names[i], axs[i])
    
    plt.tight_layout() # To prevent figures from overlapping
    plt.show()

def run_full_test(alpha, training_data, training_labels, test_data, test_labels):
    # Train and predict
    weight_matrix, _ = train_linear_classifier(training_data, training_labels, alpha, iterations)
    print(weight_matrix)
    
    training_predictions = predict(training_data, weight_matrix)
    test_predictions = predict(test_data, weight_matrix)
    
    # Convert labels to class labels
    training_predicted_labels = np.argmax(training_predictions, axis=1)
    training_true_labels = np.argmax(training_labels, axis=1)
    test_predicted_labels = np.argmax(test_predictions, axis=1)
    test_true_labels = np.argmax(test_labels, axis=1)
    
    # Calculate error rates
    training_accuracy = np.mean(training_predicted_labels == training_true_labels)
    error_rate_train = 1 - training_accuracy
    print(f"Training error rate: {error_rate_train * 100:.2f}%")
    
    test_accuracy = np.mean(test_predicted_labels == test_true_labels)
    test_error_rate = 1 - test_accuracy
    print(f"Testing error rate: {test_error_rate * 100:.2f}%")
    
    # Confusion matrices
    training_confusion_matrix = make_confusion_matrix(training_true_labels, training_predicted_labels)
    test_confusion_matrix = make_confusion_matrix(test_true_labels, test_predicted_labels)
    
    plot_confusion_matrices(training_confusion_matrix, "Confusion Matrix for Training Data", test_confusion_matrix, "Confusion Matrix for Test Data")

### Task 1 ###
def task1b():
    iris_data = np.loadtxt("iris_classification/classes/iris_data", delimiter=",", usecols=(0, 1, 2, 3))
    training_data, training_labels = get_training_data(iris_data, 0)
    
    alphas = [0.0001, 0.001, 0.01, 0.1, 1]

    plt.figure(figsize=(10, 6))
    
    for alpha in alphas:
        weight_matrix, losses = train_linear_classifier(training_data, training_labels, alpha, iterations)
        print(weight_matrix)
        plt.plot(range(iterations), losses, label=f'alpha={alpha}')

    plt.title("Mean Square Error")
    plt.xlabel("Iterations")
    plt.ylabel("Mean square error")
    plt.legend()
    plt.grid()
    plt.show()
    
def task1c():
    iris_data = np.loadtxt("iris_classification/classes/iris_data", delimiter=",", usecols=(0, 1, 2, 3))
    alpha = 0.01
    training_data, training_labels = get_training_data(iris_data, 0)
    test_data, test_labels = get_test_data(iris_data, 30)
    
    run_full_test(alpha, training_data, training_labels, test_data, test_labels)
    
def task1d():
    iris_data = np.loadtxt("iris_classification/classes/iris_data", delimiter=",", usecols=(0, 1, 2, 3))
    alpha = 0.01
    training_data, training_labels = get_training_data(iris_data, 20)
    test_data, test_labels = get_test_data(iris_data, 0)
        
    run_full_test(alpha, training_data, training_labels, test_data, test_labels)

### Task 2 ###
def task2():
    iris_data = np.loadtxt("iris_classification/classes/iris_data", delimiter=",", usecols=(0, 1, 2, 3))
    alpha = 0.01
    
    plot_all_histograms(iris_data)

    # Remove the most overlapping feature
    iris_data = np.loadtxt("iris_classification/classes/iris_data", delimiter=",", usecols=(0, 2, 3))
    training_data, training_labels = get_training_data(iris_data, 0)
    test_data, test_labels = get_test_data(iris_data, 30)

    run_full_test(alpha, training_data, training_labels, test_data, test_labels)

    # Remove the 2 most overlapping features
    iris_data = np.loadtxt("iris_classification/classes/iris_data", delimiter=",", usecols=(2, 3))
    training_data, training_labels = get_training_data(iris_data, 0)
    test_data, test_labels = get_test_data(iris_data, 30)

    run_full_test(alpha, training_data, training_labels, test_data, test_labels)

    # Remove the 3 most overlapping features
    iris_data = np.loadtxt("iris_classification/classes/iris_data", delimiter=",", usecols=(3))
    iris_data = np.expand_dims(iris_data, axis=1) # Needs to be 2D for the classifier
    training_data, training_labels = get_training_data(iris_data, 0)
    test_data, test_labels = get_test_data(iris_data, 30)

    run_full_test(alpha, training_data, training_labels, test_data, test_labels)

### Call the tasks ###
task1b()
task1c()
task1d()
task2()