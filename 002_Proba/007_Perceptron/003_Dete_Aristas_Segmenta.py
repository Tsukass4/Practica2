"""
Algoritmo 84: Detección de Aristas y Segmentación
Detectar bordes y segmentar regiones en imágenes.
"""
import numpy as np
class FiltrosImagen:
    @staticmethod
    def _to_grayscale(im):
        im = np.asarray(im)
        if im.ndim == 3:
            # convertir RGB a gris por promedio (si es color)
            return im.mean(axis=2)
        return im

    @staticmethod
    def convolucion(imagen, kernel):
        imagen = FiltrosImagen._to_grayscale(imagen).astype(float)
        kernel = np.asarray(kernel, dtype=float)
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        imagen_p = np.pad(imagen, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        kernel_flipped = np.flipud(np.fliplr(kernel))
        out = np.zeros_like(imagen)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = np.sum(imagen_p[i:i+kh, j:j+kw] * kernel_flipped)
        return out

    @staticmethod
    def filtro_gaussiano(imagen, sigma=1.0, size=None):
        imagen = FiltrosImagen._to_grayscale(imagen).astype(float)
        if size is None:
            size = int(2 * np.ceil(3 * sigma) + 1)
        ax = np.arange(-size//2 + 1, size//2 + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return FiltrosImagen.convolucion(imagen, kernel)

class DeteccionAristas:
    @staticmethod
    def sobel(imagen):
        # Operador de Sobel
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Convolución
        grad_x = FiltrosImagen.convolucion(imagen, Gx)
        grad_y = FiltrosImagen.convolucion(imagen, Gy)
        
        # Magnitud del gradiente
        magnitud = np.sqrt(grad_x**2 + grad_y**2)
        direccion = np.arctan2(grad_y, grad_x)
        
        return magnitud, direccion
    
    @staticmethod
    def canny(imagen, umbral_bajo=50, umbral_alto=150):
        # Algoritmo de Canny (simplificado)
        # 1. Suavizado gaussiano
        imagen_suave = FiltrosImagen.filtro_gaussiano(imagen, sigma=1.4)
        
        # 2. Gradientes
        magnitud, direccion = DeteccionAristas.sobel(imagen_suave)
        
        # 3. Supresión no-máxima
        # (simplificado)
        
        # 4. Umbralización por histéresis
        aristas = np.zeros_like(magnitud)
        aristas[magnitud > umbral_alto] = 255
        aristas[(magnitud > umbral_bajo) & (magnitud <= umbral_alto)] = 128
        
        return aristas

class Segmentacion:
    @staticmethod
    def umbralizado(imagen, umbral):
        return (imagen > umbral).astype(np.uint8) * 255
    
    @staticmethod
    def otsu(imagen):
        # Método de Otsu para umbralización automática
        histograma, bins = np.histogram(imagen.flatten(), bins=256, range=[0, 256])
        
        mejor_umbral = 0
        mejor_varianza = 0
        
        for t in range(1, 256):
            w0 = histograma[:t].sum()
            w1 = histograma[t:].sum()
            
            if w0 == 0 or w1 == 0:
                continue
            
            mu0 = np.sum(np.arange(t) * histograma[:t]) / w0
            mu1 = np.sum(np.arange(t, 256) * histograma[t:]) / w1
            
            varianza = w0 * w1 * (mu0 - mu1) ** 2
            
            if varianza > mejor_varianza:
                mejor_varianza = varianza
                mejor_umbral = t
        
        return mejor_umbral

print("Detección de Aristas:")
print("- Sobel: Gradientes en x e y")
print("- Canny: Detector de aristas robusto")
print("\nSegmentación:")
print("- Umbralización: Separar regiones por intensidad")
print("- Otsu: Umbral automático óptimo")
