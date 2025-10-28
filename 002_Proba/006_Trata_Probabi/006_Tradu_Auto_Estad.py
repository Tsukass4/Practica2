"""
Algoritmo 81: Traducción Automática Estadística
Traducción basada en modelos probabilísticos.
"""
from collections import defaultdict

class TraductorEstadistico:
    def __init__(self):
        self.tabla_traduccion = defaultdict(lambda: defaultdict(float))
        self.modelo_lenguaje = defaultdict(lambda: defaultdict(int))
    
    def entrenar(self, pares_paralelos):
        # Entrenar tabla de traducción
        for origen, destino in pares_paralelos:
            palabras_origen = origen.split()
            palabras_destino = destino.split()
            
            for po in palabras_origen:
                for pd in palabras_destino:
                    self.tabla_traduccion[po][pd] += 1
        
        # Normalizar
        for po in self.tabla_traduccion:
            total = sum(self.tabla_traduccion[po].values())
            for pd in self.tabla_traduccion[po]:
                self.tabla_traduccion[po][pd] /= total
    
    def traducir(self, oracion):
        palabras = oracion.split()
        traduccion = []
        
        for palabra in palabras:
            if palabra in self.tabla_traduccion:
                # Elegir traducción más probable
                mejor = max(self.tabla_traduccion[palabra].items(), 
                          key=lambda x: x[1])
                traduccion.append(mejor[0])
            else:
                traduccion.append(palabra)
        
        return ' '.join(traduccion)

# Ejemplo
pares = [
    ("hello world", "hola mundo"),
    ("hello friend", "hola amigo"),
    ("good morning", "buenos días")
]
traductor = TraductorEstadistico()
traductor.entrenar(pares)
print("Traducción Automática Estadística")
print(f"Traducción: {traductor.traducir('hello world')}")
