from .layers import Layer

class MLP:
    def __init__(self, layers):
        self.layers = layers
    
    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output