"""
Algoritmo 67: Aprendizaje Profundo (Deep Learning)
Redes neuronales profundas con múltiples capas ocultas.
"""
import numpy as np

class RedNeuronalProfunda:
    def __init__(self, arquitectura):
        self.arquitectura = arquitectura
        self.pesos = []
        self.sesgos = []
        
        # Inicializar pesos
        for i in range(len(arquitectura) - 1):
            W = np.random.randn(arquitectura[i], arquitectura[i+1]) * 0.01
            b = np.zeros((1, arquitectura[i+1]))
            self.pesos.append(W)
            self.sesgos.append(b)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def forward(self, X):
        activacion = X
        activaciones = [X]
        
        for i in range(len(self.pesos)):
            z = activacion @ self.pesos[i] + self.sesgos[i]
            if i < len(self.pesos) - 1:
                activacion = self.relu(z)
            else:
                activacion = self.sigmoid(z)
            activaciones.append(activacion)
        
        return activaciones
    
    def backward(self, X, y, activaciones, tasa_aprendizaje=0.01):
        m = len(X)
        deltas = [None] * len(self.pesos)
        
        # Capa de salida
        deltas[-1] = activaciones[-1] - y
        
        # Capas ocultas
        for i in range(len(self.pesos) - 2, -1, -1):
            deltas[i] = (deltas[i+1] @ self.pesos[i+1].T) * (activaciones[i+1] > 0)
        
        # Actualizar pesos
        for i in range(len(self.pesos)):
            self.pesos[i] -= tasa_aprendizaje * (activaciones[i].T @ deltas[i]) / m
            self.sesgos[i] -= tasa_aprendizaje * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def entrenar(self, X, y, epocas=1000):
        for _ in range(epocas):
            activaciones = self.forward(X)
            self.backward(X, y, activaciones)
    
    def predecir(self, X):
        return self.forward(X)[-1]

print("Deep Learning: Redes neuronales profundas")
print("Múltiples capas permiten aprender representaciones jerárquicas")
