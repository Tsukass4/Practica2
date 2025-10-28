"""
Algoritmo 76: Modelo Probabilístico del Lenguaje - Corpus
Modelos n-gram para predicción de palabras.
"""
from collections import defaultdict, Counter

class ModeloLenguaje:
    def __init__(self, n=2):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.vocabulario = set()
    
    def entrenar(self, corpus):
        for oracion in corpus:
            palabras = ['<START>'] * (self.n - 1) + oracion.split() + ['<END>']
            self.vocabulario.update(palabras)
            
            for i in range(len(palabras) - self.n + 1):
                contexto = tuple(palabras[i:i+self.n-1])
                siguiente = palabras[i+self.n-1]
                self.ngrams[contexto][siguiente] += 1
    
    def probabilidad(self, contexto, palabra):
        if contexto not in self.ngrams:
            return 0.0
        
        conteo_contexto = sum(self.ngrams[contexto].values())
        conteo_palabra = self.ngrams[contexto][palabra]
        
        # Suavizado de Laplace
        return (conteo_palabra + 1) / (conteo_contexto + len(self.vocabulario))
    
    def generar(self, longitud=10):
        contexto = tuple(['<START>'] * (self.n - 1))
        oracion = []
        
        for _ in range(longitud):
            if contexto not in self.ngrams:
                break
            
            palabras = list(self.ngrams[contexto].keys())
            pesos = list(self.ngrams[contexto].values())
            
            import random
            siguiente = random.choices(palabras, weights=pesos)[0]
            
            if siguiente == '<END>':
                break
            
            oracion.append(siguiente)
            contexto = tuple(list(contexto[1:]) + [siguiente])
        
        return ' '.join(oracion)

# Ejemplo
corpus = [
    "el gato come pescado",
    "el perro come carne",
    "el gato duerme mucho"
]
modelo = ModeloLenguaje(n=2)
modelo.entrenar(corpus)
print("Modelo de Lenguaje (bigram)")
print(f"Texto generado: {modelo.generar()}")
