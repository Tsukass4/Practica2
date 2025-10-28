"""
Algoritmo 70: Perceptrón, ADALINE y MADALINE
Primeros modelos de redes neuronales.
"""
import numpy as np

class Perceptron:
    def __init__(self, n_entradas, tasa_aprendizaje=0.1):
        self.pesos = np.random.randn(n_entradas)
        self.sesgo = np.random.randn()
        self.lr = tasa_aprendizaje
    
    def activacion(self, x):
        return 1 if x >= 0 else -1
    
    def predecir(self, x):
        z = np.dot(x, self.pesos) + self.sesgo
        return self.activacion(z)
    
    def entrenar(self, X, y, epocas=100):
        for _ in range(epocas):
            for xi, yi in zip(X, y):
                prediccion = self.predecir(xi)
                error = yi - prediccion
                self.pesos += self.lr * error * xi
                self.sesgo += self.lr * error

class ADALINE:
    def __init__(self, n_entradas, tasa_aprendizaje=0.01):
        self.pesos = np.random.randn(n_entradas)
        self.sesgo = np.random.randn()
        self.lr = tasa_aprendizaje
    
    def activacion_lineal(self, x):
        return x
    
    def activacion_salida(self, x):
        return 1 if x >= 0 else -1
    
    def entrenar(self, X, y, epocas=100):
        for _ in range(epocas):
            salida = X @ self.pesos + self.sesgo
            errores = y - salida
            self.pesos += self.lr * X.T @ errores
            self.sesgo += self.lr * errores.sum()

print("Perceptrón: Primer modelo de neurona artificial (1958)")
print("ADALINE: Adaptive Linear Neuron, usa regla delta")
print("MADALINE: Multiple ADALINE, primera red multicapa")
