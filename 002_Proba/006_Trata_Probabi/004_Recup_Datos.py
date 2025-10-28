"""
Algoritmo 79: Recuperación de Datos
Sistemas de recuperación de información (IR).
"""
import math
from collections import Counter

class SistemaRecuperacion:
    def __init__(self):
        self.documentos = []
        self.indice_invertido = {}
        self.idf = {}
    
    def indexar(self, documentos):
        self.documentos = documentos
        
        # Crear índice invertido
        for doc_id, doc in enumerate(documentos):
            palabras = doc.lower().split()
            for palabra in set(palabras):
                if palabra not in self.indice_invertido:
                    self.indice_invertido[palabra] = []
                self.indice_invertido[palabra].append(doc_id)
        
        # Calcular IDF
        N = len(documentos)
        for palabra, docs in self.indice_invertido.items():
            self.idf[palabra] = math.log(N / len(docs))
    
    def tf_idf(self, doc_id, palabra):
        doc = self.documentos[doc_id].lower()
        palabras = doc.split()
        
        # TF
        tf = palabras.count(palabra) / len(palabras)
        
        # IDF
        idf = self.idf.get(palabra, 0)
        
        return tf * idf
    
    def buscar(self, consulta, top_k=5):
        palabras_consulta = consulta.lower().split()
        
        # Calcular scores
        scores = {}
        for doc_id in range(len(self.documentos)):
            score = sum(self.tf_idf(doc_id, palabra) for palabra in palabras_consulta)
            scores[doc_id] = score
        
        # Ordenar y retornar top-k
        resultados = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return resultados[:top_k]

# Ejemplo
docs = [
    "el gato come pescado",
    "el perro come carne",
    "los gatos son animales"
]
sistema = SistemaRecuperacion()
sistema.indexar(docs)
resultados = sistema.buscar("gato come")
print("Recuperación de Información (TF-IDF)")
print(f"Resultados: {resultados}")
