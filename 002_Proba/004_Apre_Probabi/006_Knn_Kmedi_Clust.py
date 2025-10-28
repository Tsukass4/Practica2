"""
Algoritmo 65: k-NN, k-Medias y Clustering
Algoritmos de aprendizaje basados en vecindad y agrupamiento.
"""
import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def ajustar(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predecir(self, X):
        predicciones = []
        for x in X:
            # Calcular distancias
            distancias = np.linalg.norm(self.X_train - x, axis=1)
            # Encontrar k vecinos más cercanos
            k_indices = np.argsort(distancias)[:self.k]
            k_etiquetas = self.y_train[k_indices]
            # Votar
            from collections import Counter
            prediccion = Counter(k_etiquetas).most_common(1)[0][0]
            predicciones.append(prediccion)
        return np.array(predicciones)

class KMeans:
    def __init__(self, k=3):
        self.k = k
        self.centroides = None
    
    def ajustar(self, X, max_iter=100):
        # Inicializar centroides
        indices = np.random.choice(len(X), self.k, replace=False)
        self.centroides = X[indices]
        
        for _ in range(max_iter):
            # Asignar a clusters
            etiquetas = self.predecir(X)
            
            # Actualizar centroides
            nuevos_centroides = np.array([
                X[etiquetas == i].mean(axis=0) for i in range(self.k)
            ])
            
            if np.allclose(self.centroides, nuevos_centroides):
                break
            
            self.centroides = nuevos_centroides
    
    def predecir(self, X):
        distancias = np.linalg.norm(X[:, np.newaxis] - self.centroides, axis=2)
        return np.argmin(distancias, axis=1)

print("k-NN: Clasificación basada en vecinos más cercanos")
print("k-Means: Clustering por partición")
