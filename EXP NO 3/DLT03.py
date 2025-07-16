import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

n_x = X_train.shape[1]
n_h = 5
n_y = 1

W1 = np.random.randn(n_h, n_x) * 0.01
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(n_y, n_h) * 0.01
b2 = np.zeros((n_y, 1))

lr = 0.1
epochs = 1000
m = X_train.shape[0]

loss_history = []

for i in range(epochs):
    Z1 = np.dot(W1, X_train.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    loss = -np.mean(y_train.T * np.log(A2 + 1e-15) + (1 - y_train.T) * np.log(1 - A2 + 1e-15))
    loss_history.append(loss)

    dZ2 = A2 - y_train.T
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * np.dot(dZ1, X_train)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

def predict(X):
    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return (A2 > 0.5).astype(int).flatten()

y_pred = predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Neural Network Accuracy:", acc)

plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
