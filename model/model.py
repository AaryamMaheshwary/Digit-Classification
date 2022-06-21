import numpy as np
import pandas as pd
import json

# TODO: Test Stochastic Gradient Descent and Mini-Batch Gradient Descent

def rand_init():
    """Initialize weights and biases to random values in range: [-0.5, 0.5)"""
    w1 = np.random.rand(256, 784) - 0.5
    b1 = np.random.rand(256, 1) - 0.5
    w2 = np.random.rand(10, 256) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2


def relu(z):
    """ReLU activation function"""
    return np.maximum(z, 0)


def relu_deriv(z):
    """Derivative of ReLU (for backpropogation)"""
    return z > 0


def softmax(z):
    """Softmax activation function"""
    return np.exp(z) / sum(np.exp(z))


def one_hot(y):
    """One-hot encode output array"""
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def forward(X, w1, b1, w2, b2):
    """Forward pass through the model given weights, biases, and inputs"""
    z1 = w1.dot(X) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def backward(X, y, z1, a1, z2, a2, w1, w2):
    """Backpropogate through network to get the gradient"""
    y = one_hot(y)
    dz2 = a2 - y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * relu_deriv(z1)
    dw1 = 1 / m * dz1.dot(X.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, step_size):
    """Update weights and biases with the gradient and step size"""
    w1 = w1 - step_size * dw1
    b1 = b1 - step_size * db1    
    w2 = w2 - step_size * dw2  
    b2 = b2 - step_size * db2    
    return w1, b1, w2, b2


def get_prediction(a2):
    """Get the model's prediction based on output layer"""
    return np.argmax(a2, 0)


def get_accuracy(prediction, y):
    """Get proportion of correct guesses"""
    return np.sum(prediction == y) / y.size


def gradient_descent(X, y, step_size, epochs):
    """"""
    w1, b1, w2, b2 = rand_init()
    for i in range(epochs):
        z1, a1, z2, a2 = forward(X, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = backward(X, y, z1, a1, z2, a2, w1, w2)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, step_size)
        # Print training update
        print("Epoch: {}".format(i+1))
        train_prediction = get_prediction(a2)
        accuracy = get_accuracy(train_prediction, y)
        print("Train Accuracy: {:4.2f}%".format(accuracy*100))
        print("-" * 22)
    return w1, b1, w2, b2


# Read data and shuffle
data = pd.read_csv("model/data.csv")
data = np.array(data)
np.random.shuffle(data)

# Split Train and Test data (80-20 split)
m, n = data.shape
train_data = data[0: int(0.8*m)].T
test_data = data[int(0.8*m): m].T

# Separate X and y in data
X_train = train_data[1:n]/255
y_train = train_data[0]
X_test = test_data[1:n]/255
y_test = test_data[0]

# Do gradient descent
w1, b1, w2, b2 = gradient_descent(X_train, y_train, 0.10, 500)

# Get accuracy of model on test data
_, _, _, a2 = forward(X_test, w1, b1, w2, b2)
test_prediction = get_prediction(a2)
accuracy = get_accuracy(test_prediction, y_test)
print("Model Accuracy: {:4.2f}%".format(accuracy*100))

# Export model parameters to JSON file
model_params = {
    "w1": w1.tolist(),
    "b1": b1.tolist(),
    "w2": w2.tolist(),
    "b2": b2.tolist(),
}
with open('model/model_params.json', 'w') as f:
    json.dump(model_params, f)
