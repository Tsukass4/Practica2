"""
Algoritmo 74: Mapas Autoorganizados de Kohonen (SOM)
Red neuronal no supervisada para reducción de dimensionalidad.
"""
import numpy as np

class SOM:
    def __init__(self, ancho, alto, dim_entrada):
        self.ancho = ancho
        self.alto = alto
        self.dim_entrada = dim_entrada
        self.pesos = np.random.rand(ancho, alto, dim_entrada)
    
    def encontrar_bmu(self, x):
        # Best Matching Unit
        distancias = np.linalg.norm(self.pesos - x, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distancias), distancias.shape)
        return bmu_idx
    
    def actualizar_pesos(self, x, bmu_idx, lr, radio):
        for i in range(self.ancho):
            for j in range(self.alto):
                # Distancia a BMU
                dist = np.sqrt((i - bmu_idx[0])**2 + (j - bmu_idx[1])**2)
                
                if dist <= radio:
                    # Función de vecindad
                    influencia = np.exp(-(dist**2) / (2 * radio**2))
                    
                    # Actualizar pesos
                    self.pesos[i, j] += lr * influencia * (x - self.pesos[i, j])
    
    def entrenar(self, datos, epocas=100):
        lr_inicial = 0.1
        radio_inicial = max(self.ancho, self.alto) / 2
        
        for epoca in range(epocas):
            lr = lr_inicial * (1 - epoca / epocas)
            radio = radio_inicial * (1 - epoca / epocas)
            
            for x in datos:
                bmu_idx = self.encontrar_bmu(x)
                self.actualizar_pesos(x, bmu_idx, lr, radio)

print("SOM (Kohonen): Mapas autoorganizados")
print("Preserva topología de datos de alta dimensión en mapa 2D")
print("Aplicaciones: Visualización, clustering, reducción de dimensionalidad")
