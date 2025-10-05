import numpy as np

# -------------------------
# Funciones de activación
# -------------------------
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------------
# Clase Layer
# -------------------------
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

# -------------------------
# Clase MLP
# -------------------------
class MLP:
    def __init__(self, layers):
        self.layers = layers
    
    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

# -------------------------
# Función principal
# -------------------------
def main():
    # Datos de ejemplo: 5 muestras, 3 características
    X = np.random.rand(5, 3)
    
    # Crear un MLP simple: 2 capas
    layers = [
        Layer(n_inputs=3, n_neurons=4, activation='relu'),
        Layer(n_inputs=4, n_neurons=2, activation='sigmoid')
    ]
    
    model = MLP(layers)
    
    # Hacer predicciones
    output = model.predict(X)
    print("Salida del MLP:\n", output)

# -------------------------
# Ejecutar si es main
# -------------------------
if __name__ == "__main__":
    main()