import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_weights(n_features):
    w = np.zeros(n_features)
    b = 0
    return w, b

def compute_loss(y, y_hat):
    m = y.shape[0]
    return -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))

def train(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w, b = initialize_weights(n_features)
    for i in range(epochs):
        linear_model = np.dot(X, w) + b
        y_hat = sigmoid(linear_model)
        dw = np.dot(X.T, (y_hat - y)) / n_samples
        db = np.sum(y_hat - y) / n_samples
        w -= lr * dw
        b -= lr * db
    return w, b

def predict(X, w, b):
    y_hat = sigmoid(np.dot(X, w) + b)
    return (y_hat >= 0.5).astype(int)

w, b = train(X_train, y_train, lr=0.1, epochs=1000)
y_pred = predict(X_test, w, b)
acc = accuracy_score(y_test, y_pred)

print("Custom Logistic Regression Accuracy:", acc)
