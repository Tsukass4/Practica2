"""
Algoritmo 89: Movimiento
Análisis de movimiento y flujo óptico en secuencias de video.
"""
import numpy as np

class AnalisisMovimiento:
    @staticmethod
    def diferencia_frames(frame1, frame2):
        # Detección de movimiento simple
        diferencia = np.abs(frame1.astype(float) - frame2.astype(float))
        umbral = 30
        mascara_movimiento = diferencia > umbral
        return mascara_movimiento
    
    @staticmethod
    def flujo_optico_lucas_kanade(frame1, frame2, puntos):
        # Método de Lucas-Kanade para flujo óptico (simplificado)
        ventana = 5
        flujo = []
        
        for x, y in puntos:
            # Extraer ventana
            ventana1 = frame1[y-ventana:y+ventana+1, x-ventana:x+ventana+1]
            
            # Calcular gradientes
            Ix = np.gradient(ventana1, axis=1)
            Iy = np.gradient(ventana1, axis=0)
            It = frame2[y-ventana:y+ventana+1, x-ventana:x+ventana+1] - ventana1
            
            # Resolver sistema de ecuaciones
            # [u, v]^T = -(A^T A)^-1 A^T b
            Ix_flat = Ix.flatten()
            Iy_flat = Iy.flatten()
            It_flat = It.flatten()
            
            A = np.column_stack([Ix_flat, Iy_flat])
            b = -It_flat
            
            try:
                velocidad = np.linalg.lstsq(A, b, rcond=None)[0]
                flujo.append(velocidad)
            except:
                flujo.append([0, 0])
        
        return np.array(flujo)
    
    @staticmethod
    def background_subtraction(frames, metodo='media'):
        # Sustracción de fondo
        if metodo == 'media':
            fondo = np.mean(frames, axis=0)
        elif metodo == 'mediana':
            fondo = np.median(frames, axis=0)
        else:
            fondo = frames[0]
        
        # Detectar objetos en movimiento
        objetos_movimiento = []
        for frame in frames:
            diferencia = np.abs(frame - fondo)
            mascara = diferencia > 30
            objetos_movimiento.append(mascara)
        
        return fondo, objetos_movimiento
    
    @staticmethod
    def tracking_objetos(detecciones_frames):
        # Seguimiento simple de objetos entre frames
        # Usar centroide y distancia mínima
        
        tracks = []
        for i, detecciones in enumerate(detecciones_frames):
            if i == 0:
                # Inicializar tracks
                tracks = [[d] for d in detecciones]
            else:
                # Asociar detecciones con tracks existentes
                for deteccion in detecciones:
                    # Encontrar track más cercano
                    distancias = [
                        np.linalg.norm(np.array(deteccion) - np.array(track[-1]))
                        for track in tracks
                    ]
                    
                    if distancias:
                        idx_min = np.argmin(distancias)
                        if distancias[idx_min] < 50:  # Umbral de distancia
                            tracks[idx_min].append(deteccion)
                        else:
                            tracks.append([deteccion])
                    else:
                        tracks.append([deteccion])
        
        return tracks

print("Análisis de Movimiento:")
print("- Diferencia de frames: Detección simple")
print("- Flujo óptico: Estimar velocidad de píxeles")
print("  • Lucas-Kanade: Método local")
print("  • Horn-Schunck: Método global")
print("- Background subtraction: Detectar objetos en movimiento")
print("- Tracking: Seguir objetos a través del tiempo")
print("\nAplicaciones:")
print("- Vigilancia")
print("- Análisis deportivo")
print("- Vehículos autónomos")
