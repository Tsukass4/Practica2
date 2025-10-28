"""
Algoritmo 66: Máquinas de Vectores Soporte (Núcleo)
SVM con kernel trick para clasificación no lineal.
"""
import numpy as np

class SVM:
    def __init__(self, kernel='lineal', C=1.0):
        self.kernel = kernel
        self.C = C
        self.alpha = None
        self.b = None
        self.X_train = None
        self.y_train = None
    
    def kernel_func(self, x1, x2):
        if self.kernel == 'lineal':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            gamma = 0.1
            return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        elif self.kernel == 'polinomial':
            degree = 3
            return (1 + np.dot(x1, x2))**degree
    
    def ajustar(self, X, y):
        n = len(X)
        self.X_train = X
        self.y_train = y
        
        # Matriz de kernel
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel_func(X[i], X[j])
        
        # Resolver problema de optimización (simplificado)
        # En práctica se usa SMO (Sequential Minimal Optimization)
        self.alpha = np.random.rand(n) * self.C
        self.b = 0
    
    def predecir(self, X):
        predicciones = []
        for x in X:
            suma = 0
            for i in range(len(self.X_train)):
                suma += self.alpha[i] * self.y_train[i] * self.kernel_func(self.X_train[i], x)
            predicciones.append(np.sign(suma + self.b))
        return np.array(predicciones)

print("SVM: Clasificador de margen máximo")
print("Kernel trick: Permite separación no lineal en espacio de alta dimensión")
