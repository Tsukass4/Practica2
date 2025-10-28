"""
Algoritmo 82: Gráficos por Computador
Fundamentos de renderizado y geometría computacional.
"""
import numpy as np

class GraficosComputador:
    @staticmethod
    def transformacion_2d(puntos, matriz):
        # Aplicar transformación matricial
        return puntos @ matriz.T
    
    @staticmethod
    def rotacion_2d(angulo):
        c, s = np.cos(angulo), np.sin(angulo)
        return np.array([[c, -s], [s, c]])
    
    @staticmethod
    def escalado_2d(sx, sy):
        return np.array([[sx, 0], [0, sy]])
    
    @staticmethod
    def traslacion_2d(tx, ty):
        # Coordenadas homogéneas
        return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    
    @staticmethod
    def proyeccion_perspectiva(puntos_3d, distancia_focal):
        # Proyección perspectiva simple
        proyectados = []
        for x, y, z in puntos_3d:
            if z != 0:
                x_proj = distancia_focal * x / z
                y_proj = distancia_focal * y / z
                proyectados.append([x_proj, y_proj])
        return np.array(proyectados)

print("Gráficos por Computador: Transformaciones geométricas")
print("- Rotación, escalado, traslación")
print("- Proyección perspectiva")
