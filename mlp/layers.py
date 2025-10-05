import numpy as np

# Funciones de activación
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Clase Layer
class Layer:
    def __init__(self, n_inputs, n_neurons, activation='relu'):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))
        
        if activation == 'relu':
            self.activation = relu
        elif activation == 'sigmoid':
            self.activation = sigmoid
        else:
            raise ValueError("Activación no soportada")
    
    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.biases
        return self.activation(z)