"""
Algoritmo 52: Procesos Estacionarios
Proceso estocástico cuyas propiedades no cambian en el tiempo.
P(Xt | Xt-1) = P(Xs | Xs-1) para todo t, s
"""
class ProcesoEstacionario:
    def __init__(self, matriz_transicion):
        self.T = matriz_transicion
    
    def prob_transicion(self, estado_actual, estado_siguiente):
        return self.T[estado_actual][estado_siguiente]
    
    def distribucion_estacionaria(self):
        # Resolver πT = π
        import numpy as np
        A = np.array(self.T).T - np.eye(len(self.T))
        A = np.vstack([A, np.ones(len(self.T))])
        b = np.zeros(len(self.T) + 1)
        b[-1] = 1
        return np.linalg.lstsq(A, b, rcond=None)[0]

# Ejemplo
print("Proceso Estacionario: Propiedades invariantes en el tiempo")
T = [[0.7, 0.3], [0.4, 0.6]]
proceso = ProcesoEstacionario(T)
print(f"Distribución estacionaria: {proceso.distribucion_estacionaria()}")
