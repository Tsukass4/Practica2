"""
Algoritmo 80: Extracción de Información
Extracción de entidades, relaciones y eventos de texto.
"""
import re

class ExtractorInformacion:
    def __init__(self):
        self.patrones_entidades = {
            'PERSONA': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'FECHA': r'\d{1,2}/\d{1,2}/\d{4}',
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
    
    def extraer_entidades(self, texto):
        entidades = {}
        for tipo, patron in self.patrones_entidades.items():
            matches = re.findall(patron, texto)
            if matches:
                entidades[tipo] = matches
        return entidades
    
    def extraer_relaciones(self, texto):
        # Patrón simple: X verbo Y
        patron = r'([A-Z][a-z]+) (es|trabaja en|vive en) ([A-Z][a-z]+)'
        relaciones = re.findall(patron, texto)
        return [(sujeto, verbo, objeto) for sujeto, verbo, objeto in relaciones]

# Ejemplo
texto = "Juan Pérez trabaja en Microsoft. Su email es juan@microsoft.com. Fecha: 01/01/2024"
extractor = ExtractorInformacion()
entidades = extractor.extraer_entidades(texto)
print("Extracción de Información:")
print(f"Entidades: {entidades}")
