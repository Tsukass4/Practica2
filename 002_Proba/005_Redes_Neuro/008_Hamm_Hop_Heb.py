"""
Algoritmo 75: Redes de Hamming, Hopfield, Hebb, Boltzmann
Arquitecturas clásicas de redes neuronales.
"""
import numpy as np

class RedHopfield:
    def __init__(self, n):
        self.n = n
        self.W = np.zeros((n, n))
    
    def entrenar(self, patrones):
        # Regla de Hebb
        for patron in patrones:
            self.W += np.outer(patron, patron)
        
        # Eliminar autoconexiones
        np.fill_diagonal(self.W, 0)
        self.W /= len(patrones)
    
    def recuperar(self, patron, max_iter=100):
        estado = patron.copy()
        for _ in range(max_iter):
            for i in range(self.n):
                suma = np.dot(self.W[i], estado)
                estado[i] = 1 if suma >= 0 else -1
        return estado

class RedHamming:
    def __init__(self, patrones):
        self.patrones = np.array(patrones)
        self.n = len(patrones[0])
    
    def reconocer(self, entrada):
        # Calcular similitud con cada patrón
        similitudes = [np.sum(entrada == p) for p in self.patrones]
        return np.argmax(similitudes)

class MaquinaBoltzmann:
    def __init__(self, n_visible, n_oculta):
        self.n_v = n_visible
        self.n_h = n_oculta
        self.W = np.random.randn(n_visible, n_oculta) * 0.01
    
    def energia(self, v, h):
        return -np.dot(v, np.dot(self.W, h))
    
    def muestrear_oculta(self, v, temperatura=1.0):
        prob = 1 / (1 + np.exp(-np.dot(v, self.W) / temperatura))
        return (np.random.rand(self.n_h) < prob).astype(int)
    
    def muestrear_visible(self, h, temperatura=1.0):
        prob = 1 / (1 + np.exp(-np.dot(self.W, h) / temperatura))
        return (np.random.rand(self.n_v) < prob).astype(int)

print("Redes Neuronales Clásicas:")
print("- Hopfield: Memoria asociativa, recupera patrones")
print("- Hamming: Reconocimiento de patrones por similitud")
print("- Hebb: Regla de aprendizaje 'neuronas que disparan juntas, se conectan'")
print("- Boltzmann: Red estocástica basada en energía")
