import numpy as np
from mlp.layers import Layer
from mlp.network import MLP

def main():
    # Datos de ejemplo: 5 muestras, 3 características
    X = np.random.rand(5, 3)
    
    # Crear MLP con 2 capas
    layers = [
        Layer(n_inputs=3, n_neurons=4, activation='relu'),
        Layer(n_inputs=4, n_neurons=2, activation='sigmoid')
    ]
    
    model = MLP(layers)
    
    # Hacer predicciones
    output = model.predict(X)
    print("Salida del MLP:\n", output)

if __name__ == "__main__":
    main()