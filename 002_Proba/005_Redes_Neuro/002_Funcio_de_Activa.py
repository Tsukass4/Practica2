"""
Algoritmo 69: Funciones de Activación
Funciones no lineales que permiten a las redes aprender patrones complejos.
"""
import numpy as np
import matplotlib.pyplot as plt

class FuncionesActivacion:
    @staticmethod
    def escalon(x):
        return np.where(x >= 0, 1, 0)
    
    @staticmethod
    def sigmoide(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

# Ejemplo
print("Funciones de Activación:")
print("- Escalón: Binaria, no diferenciable")
print("- Sigmoide: σ(x) = 1/(1+e^-x), rango (0,1)")
print("- Tanh: tanh(x), rango (-1,1)")
print("- ReLU: max(0,x), evita vanishing gradient")
print("- Softmax: Para clasificación multiclase")
