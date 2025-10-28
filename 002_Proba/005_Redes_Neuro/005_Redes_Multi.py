"""
Algoritmo 72: Redes Multicapa
Redes con capas ocultas que pueden aprender funciones no lineales.
"""
import numpy as np

class RedMulticapa:
    def __init__(self, capas):
        self.capas = capas
        self.pesos = []
        self.sesgos = []
        
        for i in range(len(capas) - 1):
            W = np.random.randn(capas[i], capas[i+1]) * 0.1
            b = np.zeros((1, capas[i+1]))
            self.pesos.append(W)
            self.sesgos.append(b)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivada(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        self.activaciones = [X]
        self.z_valores = []
        
        activacion = X
        for W, b in zip(self.pesos, self.sesgos):
            z = activacion @ W + b
            self.z_valores.append(z)
            activacion = self.sigmoid(z)
            self.activaciones.append(activacion)
        
        return activacion
    
    def backward(self, X, y, lr=0.1):
        m = len(X)
        
        # Gradiente de la capa de salida
        delta = self.activaciones[-1] - y
        
        # Retropropagar
        for i in range(len(self.pesos) - 1, -1, -1):
            grad_W = self.activaciones[i].T @ delta / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            self.pesos[i] -= lr * grad_W
            self.sesgos[i] -= lr * grad_b
            
            if i > 0:
                delta = (delta @ self.pesos[i].T) * self.sigmoid_derivada(self.z_valores[i-1])
    
    def entrenar(self, X, y, epocas=1000):
        for _ in range(epocas):
            self.forward(X)
            self.backward(X, y)

print("Redes Multicapa: Pueden aprender funciones no lineales")
print("Capas ocultas permiten representaciones jerárquicas")
