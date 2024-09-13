# importing required libraries 
import pickle
import numpy as np
import pandas as pd

# Loading the dataset where features are :  'area', 'bedrooms', 'bathrooms', 'stories' and 'mainroad' and Target is : 'price'
data=pd.read_csv('housing.csv')

X = data[['area', 'bedrooms', 'bathrooms', 'stories','mainroad']].values
y = data['price'].values

# Convert 'mainroad' to binary (0 or 1)
X[:, 4] = np.where(X[:, 4] == 'yes', 1, 0)

# Ensure that X and y are numeric
X = X.astype(float)
y = y.astype(float)

# Normalize the data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std[X_std == 0] = 1
X = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y_std = y_std if y_std != 0 else 1
y = (y - y_mean) / y_std
y = y.reshape(-1, 1)

# Neural Network Architecture
input_size = X.shape[1]
hidden_size = 10
output_size = 1

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Activation function (ReLU)
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

# Forward propagation
def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    return Z1, A1, Z2

# Cost function (Mean Squared Error)
def compute_cost(y, y_pred):
    m = y.shape[0]
    cost = (1/m) * np.sum((y_pred - y) ** 2)
    return cost

# Backpropagation
def backpropagation(X, y, Z1, A1, Z2):
    m = X.shape[0]
    dZ2 = (Z2 - y) / m
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# Training function
def train_model(X, y, learning_rate=0.01, num_iterations=1000):
    global W1, b1, W2, b2
    for i in range(num_iterations):
        Z1, A1, Z2 = forward_propagation(X)
        cost = compute_cost(y, Z2)
        dW1, db1, dW2, db2 = backpropagation(X, y, Z1, A1, Z2)
        
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost}")

# Prediction
def predict(X):
    _, _, Z2 = forward_propagation(X)
    return Z2

# Denormalize the predictions
def denormalize(y_pred, y_mean, y_std):
    return (y_pred * y_std) + y_mean

# Split data into training and test sets (80% train, 20% test)
def train_test_split(X, y, test_size=0.2):
    m = X.shape[0]
    indices = np.random.permutation(m)
    test_set_size = int(m * test_size)
    
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
train_model(X_train, y_train)

# Save the model weights
model_data = {
    'W1': W1,
    'b1': b1,
    'W2': W2,
    'b2': b2,
    'X_mean': X_mean,
    'X_std': X_std,
    'y_mean': y_mean,
    'y_std': y_std
}

with open('model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print("Model saved successfully.")
