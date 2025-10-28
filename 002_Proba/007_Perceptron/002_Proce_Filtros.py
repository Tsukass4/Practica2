"""
Algoritmo 83: Preprocesado - Filtros
Filtros para procesamiento de imágenes.
"""
import numpy as np

class FiltrosImagen:
    @staticmethod
    def filtro_media(imagen, tamano=3):
        # Filtro de media (blur)
        pad = tamano // 2
        imagen_pad = np.pad(imagen, pad, mode='edge')
        resultado = np.zeros_like(imagen)
        
        for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                ventana = imagen_pad[i:i+tamano, j:j+tamano]
                resultado[i, j] = np.mean(ventana)
        
        return resultado
    
    @staticmethod
    def filtro_mediana(imagen, tamano=3):
        # Filtro de mediana (reduce ruido sal y pimienta)
        pad = tamano // 2
        imagen_pad = np.pad(imagen, pad, mode='edge')
        resultado = np.zeros_like(imagen)
        
        for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                ventana = imagen_pad[i:i+tamano, j:j+tamano]
                resultado[i, j] = np.median(ventana)
        
        return resultado
    
    @staticmethod
    def filtro_gaussiano(imagen, sigma=1.0):
        # Filtro gaussiano
        tamano = int(6 * sigma + 1)
        if tamano % 2 == 0:
            tamano += 1
        
        # Crear kernel gaussiano
        ax = np.arange(-tamano // 2 + 1., tamano // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        
        # Aplicar convolución
        return FiltrosImagen.convolucion(imagen, kernel)
    
    @staticmethod
    def convolucion(imagen, kernel):
        pad = kernel.shape[0] // 2
        imagen_pad = np.pad(imagen, pad, mode='edge')
        resultado = np.zeros_like(imagen)
        
        for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                ventana = imagen_pad[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                resultado[i, j] = np.sum(ventana * kernel)
        
        return resultado

print("Filtros de Imagen:")
print("- Media: Suavizado")
print("- Mediana: Reducción de ruido")
print("- Gaussiano: Suavizado ponderado")
