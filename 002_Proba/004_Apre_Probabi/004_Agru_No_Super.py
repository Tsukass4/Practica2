"""
Algoritmo 63: Agrupamiento No Supervisado
Clustering sin etiquetas: K-Means, clustering jerárquico, DBSCAN.
"""
import numpy as np

class KMeans:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroides = None
    
    def ajustar(self, X):
        # Inicializar centroides aleatoriamente
        n = len(X)
        indices = np.random.choice(n, self.k, replace=False)
        self.centroides = X[indices]
        
        for _ in range(self.max_iter):
            # Asignar puntos a clusters
            clusters = self.asignar_clusters(X)
            
            # Actualizar centroides
            nuevos_centroides = np.array([
                X[clusters == i].mean(axis=0) if np.any(clusters == i) else self.centroides[i]
                for i in range(self.k)
            ])
            
            # Verificar convergencia
            if np.allclose(self.centroides, nuevos_centroides):
                break
            
            self.centroides = nuevos_centroides
    
    def asignar_clusters(self, X):
        distancias = np.linalg.norm(X[:, np.newaxis] - self.centroides, axis=2)
        return np.argmin(distancias, axis=1)
    
    def predecir(self, X):
        return self.asignar_clusters(X)

print("Agrupamiento No Supervisado: Encontrar estructura en datos sin etiquetas")
print("K-Means: Particiona datos en k clusters minimizando varianza intra-cluster")
