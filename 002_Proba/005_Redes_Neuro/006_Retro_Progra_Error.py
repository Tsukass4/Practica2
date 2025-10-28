"""
Algoritmo 73: Retropropagación del Error
Algoritmo para entrenar redes neuronales multicapa.
"""
import numpy as np

class Retropropagacion:
    @staticmethod
    def calcular_gradientes(red, X, y):
        # Forward pass
        activaciones = red.forward(X)
        
        # Backward pass
        gradientes_W = []
        gradientes_b = []
        
        # Error de salida
        delta = activaciones[-1] - y
        
        # Retropropagar a través de las capas
        for i in range(len(red.pesos) - 1, -1, -1):
            # Gradientes
            grad_W = activaciones[i].T @ delta
            grad_b = np.sum(delta, axis=0, keepdims=True)
            
            gradientes_W.insert(0, grad_W)
            gradientes_b.insert(0, grad_b)
            
            # Propagar error a capa anterior
            if i > 0:
                delta = (delta @ red.pesos[i].T) * red.derivada_activacion(activaciones[i])
        
        return gradientes_W, gradientes_b
    
    @staticmethod
    def actualizar_pesos(red, gradientes_W, gradientes_b, lr=0.01):
        for i in range(len(red.pesos)):
            red.pesos[i] -= lr * gradientes_W[i]
            red.sesgos[i] -= lr * gradientes_b[i]

print("Retropropagación: Algoritmo clave para entrenar redes neuronales")
print("1. Forward pass: Calcular salida")
print("2. Calcular error")
print("3. Backward pass: Propagar error hacia atrás")
print("4. Actualizar pesos usando gradientes")
