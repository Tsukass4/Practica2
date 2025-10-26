"""
Algoritmo 58: Red Bayesiana Dinámica - Filtrado de Partículas
Método de Monte Carlo para filtrado en sistemas no lineales.
"""
import random
import numpy as np

class FiltradoParticulas:
    def __init__(self, n_particulas, modelo_transicion, modelo_observacion):
        self.n = n_particulas
        self.transicion = modelo_transicion
        self.observacion = modelo_observacion
        self.particulas = []
        self.pesos = []
    
    def inicializar(self, distribucion_inicial):
        self.particulas = [distribucion_inicial() for _ in range(self.n)]
        self.pesos = [1.0/self.n] * self.n
    
    def predecir(self):
        # Propagar partículas según modelo de transición
        self.particulas = [self.transicion(p) for p in self.particulas]
    
    def actualizar(self, observacion):
        # Actualizar pesos según verosimilitud
        for i, p in enumerate(self.particulas):
            self.pesos[i] *= self.observacion(p, observacion)
        
        # Normalizar pesos
        total = sum(self.pesos)
        self.pesos = [w/total for w in self.pesos]
    
    def remuestrear(self):
        # Remuestreo sistemático
        indices = random.choices(range(self.n), weights=self.pesos, k=self.n)
        self.particulas = [self.particulas[i] for i in indices]
        self.pesos = [1.0/self.n] * self.n
    
    def estimar_estado(self):
        # Estimación ponderada
        return np.average(self.particulas, weights=self.pesos, axis=0)

print("Filtrado de Partículas: Método de Monte Carlo para filtrado")
print("Útil para sistemas no lineales y no gaussianos")
