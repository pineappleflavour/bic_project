# main.py

import numpy as np
import pandas as pd
from activation_funcs import sigmoid, tanh, relu, linear
from architecture import architecture
from loss_funcs import mse, mae, crossEntropy
from preprocessing import preprocessing

# Load and Preprocess Data
data = pd.read_csv('concrete_data.csv')
X, y = preprocessing(data)

# Initialize Network Parameters
network_params = architecture()  # User-defined architecture configuration
layer_dimensions = network_params["layer_dimensions"]
learning_rate = network_params["learning_rate"]
epochs = network_params["epochs"]
activations = network_params["activations"]

# Initialize Model Weights
def initialize_weights(layer_dimensions):
    parameters = {}
    L = len(layer_dimensions)
    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(layer_dimensions[l], layer_dimensions[l-1]) * np.sqrt(2 / layer_dimensions[l-1])
        parameters[f"b{l}"] = np.zeros((layer_dimensions[l], 1))
    return parameters

parameters = initialize_weights(layer_dimensions)

# Forward Propagation Function
def forward_propagation(X, parameters, activations):
    A = X
    caches = []
    L = len(parameters) // 2
    for l in range(1, L + 1):
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        Z = np.dot(W, A) + b
        if activations[l - 1] == "relu":
            A, _ = relu(Z)
        elif activations[l - 1] == "sigmoid":
            A, _ = sigmoid(Z)
        elif activations[l - 1] == "tanh":
            A, _ = tanh(Z)
        elif activations[l - 1] == "linear":
            A, _ = linear(Z)
        caches.append((Z, A))
    return A, caches

# Backpropagation Function
def backward_propagation(X, y, parameters, caches, activations):
    grads = {}
    L = len(caches)
    m = X.shape[1]
    A_last = caches[-1][1]
    dA = A_last - y
    for l in reversed(range(1, L + 1)):
        Z, A_prev = caches[l - 1]
        W = parameters[f"W{l}"]
        if activations[l - 1] == "relu":
            _, dZ = relu(Z)
        elif activations[l - 1] == "sigmoid":
            _, dZ = sigmoid(Z)
        elif activations[l - 1] == "tanh":
            _, dZ = tanh(Z)
        elif activations[l - 1] == "linear":
            _, dZ = linear(Z)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        grads[f"dW{l}"] = dW
        grads[f"db{l}"] = db
        dA = dA_prev
    return grads

# Update Parameters Function
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters

# Training Loop
for epoch in range(epochs):
    output, caches = forward_propagation(X, parameters, activations)
    loss = mse(y, output)  # Mean squared error as an example
    grads = backward_propagation(X, y, parameters, caches, activations)
    parameters = update_parameters(parameters, grads, learning_rate)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

# Final evaluation or saving parameters 
