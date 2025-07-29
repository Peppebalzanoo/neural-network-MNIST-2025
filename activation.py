import numpy as np


# Identity function
def identity(X, derivative=False):
    """Identity activation function."""
    if not derivative:
        return X
    return np.ones_like(X)


# Sigmoid function
def sigmoid(X, derivative=False):
    """Sigmoid activation function."""
    if not derivative:
        return 1 / (1 + np.exp(-X))
    sig = sigmoid(X)
    return sig * (1 - sig)


# Tanh function
def tanh(X, derivative=False):
    """Tanh activation function."""
    if not derivative:
        return np.tanh(X)
    return 1 - np.tanh(X) ** 2


# ReLU function
def relu(X, derivative=False):
    """ReLU activation function."""
    if not derivative:
        return np.maximum(0, X)
    return np.where(X > 0, 1, 0)
