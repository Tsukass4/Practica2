"""
Algoritmo 57: Filtros de Kalman
Filtrado óptimo para sistemas lineales gaussianos.
"""
import numpy as np

class FiltroKalman:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = F  # Matriz de transición
        self.H = H  # Matriz de observación
        self.Q = Q  # Covarianza del ruido del proceso
        self.R = R  # Covarianza del ruido de observación
        self.x = x0  # Estado inicial
        self.P = P0  # Covarianza inicial
    
    def predecir(self):
        # Predicción
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def actualizar(self, z):
        # Actualización con observación z
        y = z - self.H @ self.x  # Innovación
        S = self.H @ self.P @ self.H.T + self.R  # Covarianza de innovación
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Ganancia de Kalman
        
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
    
    def filtrar(self, observaciones):
        estados = []
        for z in observaciones:
            self.predecir()
            self.actualizar(z)
            estados.append(self.x.copy())
        return estados

print("Filtro de Kalman: Filtrado óptimo para sistemas lineales gaussianos")
print("Usado en navegación, seguimiento, control")
