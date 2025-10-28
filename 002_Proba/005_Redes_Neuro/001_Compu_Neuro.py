"""
Algoritmo 68: Computación Neuronal
Fundamentos de neuronas artificiales y procesamiento distribuido.
"""
import numpy as np

class Neurona:
    def __init__(self, n_entradas):
        self.pesos = np.random.randn(n_entradas)
        self.sesgo = np.random.randn()
    
    def activacion(self, x):
        # Función escalón
        return 1 if x >= 0 else 0
    
    def forward(self, entradas):
        suma_ponderada = np.dot(entradas, self.pesos) + self.sesgo
        return self.activacion(suma_ponderada)

class CapaNeuronal:
    def __init__(self, n_entradas, n_neuronas):
        self.neuronas = [Neurona(n_entradas) for _ in range(n_neuronas)]
    
    def forward(self, entradas):
        return np.array([n.forward(entradas) for n in self.neuronas])

print("Computación Neuronal: Procesamiento inspirado en el cerebro")
print("Neurona: Σ(wi*xi) + b → f(z)")
