"""
Algoritmo 39: Probabilidad a Priori

La probabilidad a priori (o prior) es la probabilidad inicial de un evento
antes de observar cualquier evidencia. Es fundamental en razonamiento bayesiano.

Conceptos:
- P(A): Probabilidad a priori del evento A
- Estimación de priors desde datos
- Priors informativos vs no informativos
- Actualización bayesiana
"""

from typing import Dict, List
from collections import Counter
import random


class ProbabilidadPriori:
    """Manejo de probabilidades a priori"""
    
    def __init__(self):
        self.priors = {}
    
    def establecer_prior(self, evento: str, probabilidad: float):
        """Establece la probabilidad a priori de un evento"""
        if not 0 <= probabilidad <= 1:
            raise ValueError("La probabilidad debe estar entre 0 y 1")
        self.priors[evento] = probabilidad
    
    def obtener_prior(self, evento: str) -> float:
        """Obtiene la probabilidad a priori de un evento"""
        return self.priors.get(evento, 0.0)
    
    def prior_uniforme(self, eventos: List[str]):
        """Establece prior uniforme para todos los eventos"""
        prob = 1.0 / len(eventos)
        for evento in eventos:
            self.priors[evento] = prob
    
    def prior_desde_frecuencias(self, datos: List[str]):
        """Calcula priors desde frecuencias observadas"""
        conteos = Counter(datos)
        total = len(datos)
        
        for evento, conteo in conteos.items():
            self.priors[evento] = conteo / total
    
    def prior_laplace(self, eventos: List[str], datos: List[str]):
        """
        Prior de Laplace (suavizado): Agrega 1 a cada conteo
        Útil para evitar probabilidades cero
        """
        conteos = Counter(datos)
        total = len(datos) + len(eventos)  # +1 por cada evento posible
        
        for evento in eventos:
            conteo = conteos.get(evento, 0) + 1
            self.priors[evento] = conteo / total


# Ejemplo 1: Clasificación de documentos
def ejemplo_clasificacion_documentos():
    """Ejemplo de priors en clasificación de texto"""
    print("=== Probabilidad a Priori: Clasificación de Documentos ===\n")
    
    # Datos de entrenamiento: categorías de documentos
    documentos_entrenamiento = [
        "deportes", "deportes", "deportes", "deportes", "deportes",
        "politica", "politica", "politica",
        "tecnologia", "tecnologia", "tecnologia", "tecnologia",
        "entretenimiento", "entretenimiento"
    ]
    
    print(f"Total de documentos de entrenamiento: {len(documentos_entrenamiento)}")
    print(f"Distribución: {Counter(documentos_entrenamiento)}\n")
    
    # Calcular priors desde frecuencias
    modelo = ProbabilidadPriori()
    modelo.prior_desde_frecuencias(documentos_entrenamiento)
    
    print("Probabilidades a Priori (desde frecuencias):")
    for categoria in sorted(modelo.priors.keys()):
        prob = modelo.priors[categoria]
        print(f"  P({categoria}) = {prob:.3f} ({prob*100:.1f}%)")
    
    # Prior uniforme (sin información previa)
    modelo_uniforme = ProbabilidadPriori()
    categorias = list(set(documentos_entrenamiento))
    modelo_uniforme.prior_uniforme(categorias)
    
    print("\nProbabilidades a Priori (uniforme, sin información):")
    for categoria in sorted(modelo_uniforme.priors.keys()):
        prob = modelo_uniforme.priors[categoria]
        print(f"  P({categoria}) = {prob:.3f} ({prob*100:.1f}%)")
    
    # Prior de Laplace (suavizado)
    modelo_laplace = ProbabilidadPriori()
    modelo_laplace.prior_laplace(categorias, documentos_entrenamiento)
    
    print("\nProbabilidades a Priori (Laplace, suavizado):")
    for categoria in sorted(modelo_laplace.priors.keys()):
        prob = modelo_laplace.priors[categoria]
        print(f"  P({categoria}) = {prob:.3f} ({prob*100:.1f}%)")


# Ejemplo 2: Diagnóstico médico
def ejemplo_diagnostico_medico():
    """Ejemplo de priors en diagnóstico médico"""
    print("\n\n=== Probabilidad a Priori: Diagnóstico Médico ===\n")
    
    modelo = ProbabilidadPriori()
    
    # Priors basados en prevalencia de enfermedades
    # (datos de población general)
    modelo.establecer_prior("gripe", 0.05)  # 5% de la población
    modelo.establecer_prior("resfriado", 0.15)  # 15%
    modelo.establecer_prior("alergia", 0.20)  # 20%
    modelo.establecer_prior("sano", 0.60)  # 60%
    
    print("Probabilidades a Priori (prevalencia en población):")
    for enfermedad in ["gripe", "resfriado", "alergia", "sano"]:
        prob = modelo.obtener_prior(enfermedad)
        print(f"  P({enfermedad:10s}) = {prob:.3f} ({prob*100:.0f}%)")
    
    print("\nInterpretación:")
    print("  - Sin ningún síntoma, la probabilidad de estar sano es 60%")
    print("  - La alergia es más común (20%) que la gripe (5%)")
    print("  - Estos priors se actualizarán con evidencia (síntomas)")


# Ejemplo 3: Predicción del clima
def ejemplo_prediccion_clima():
    """Ejemplo de priors en predicción del clima"""
    print("\n\n=== Probabilidad a Priori: Predicción del Clima ===\n")
    
    # Datos históricos de 100 días
    clima_historico = (
        ["soleado"] * 60 +
        ["nublado"] * 25 +
        ["lluvioso"] * 15
    )
    
    random.shuffle(clima_historico)
    
    modelo = ProbabilidadPriori()
    modelo.prior_desde_frecuencias(clima_historico)
    
    print("Probabilidades a Priori (basadas en 100 días históricos):")
    for clima in ["soleado", "nublado", "lluvioso"]:
        prob = modelo.obtener_prior(clima)
        print(f"  P({clima:10s}) = {prob:.2f}")
    
    # Predicción ingenua (solo usando prior)
    print("\nPredicción para mañana (solo usando prior):")
    prediccion = max(modelo.priors.items(), key=lambda x: x[1])
    print(f"  Clima más probable: {prediccion[0]} (P = {prediccion[1]:.2f})")
    
    print("\nNota: Esta predicción ignora evidencia actual (cielo, presión, etc.)")
    print("      En la práctica, combinaríamos el prior con observaciones actuales")


# Ejemplo 4: Spam vs No Spam
def ejemplo_filtro_spam():
    """Ejemplo de priors en filtrado de spam"""
    print("\n\n=== Probabilidad a Priori: Filtro de Spam ===\n")
    
    # Datos de entrenamiento
    emails = ["spam"] * 300 + ["no_spam"] * 700
    
    modelo = ProbabilidadPriori()
    modelo.prior_desde_frecuencias(emails)
    
    print(f"Emails de entrenamiento: {len(emails)}")
    print(f"  - Spam: 300")
    print(f"  - No Spam: 700")
    
    print("\nProbabilidades a Priori:")
    print(f"  P(Spam) = {modelo.obtener_prior('spam'):.2f}")
    print(f"  P(No Spam) = {modelo.obtener_prior('no_spam'):.2f}")
    
    print("\nInterpretación:")
    print("  - Sin ver el contenido del email, hay 30% de probabilidad de spam")
    print("  - Este prior se actualizará al analizar palabras del email")
    
    # Comparar con prior uniforme
    modelo_uniforme = ProbabilidadPriori()
    modelo_uniforme.prior_uniforme(["spam", "no_spam"])
    
    print("\nPrior Uniforme (sin información previa):")
    print(f"  P(Spam) = {modelo_uniforme.obtener_prior('spam'):.2f}")
    print(f"  P(No Spam) = {modelo_uniforme.obtener_prior('no_spam'):.2f}")
    
    print("\nDiferencia:")
    print("  - Prior informativo (30/70) refleja datos reales")
    print("  - Prior uniforme (50/50) asume desconocimiento total")


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Probabilidad a Priori ===\n")
    
    ejemplo_clasificacion_documentos()
    ejemplo_diagnostico_medico()
    ejemplo_prediccion_clima()
    ejemplo_filtro_spam()
    
    print("\n" + "="*70)
    print("\nConceptos Clave:")
    print("  - Prior: Probabilidad antes de observar evidencia")
    print("  - Prior informativo: Basado en conocimiento/datos previos")
    print("  - Prior no informativo: Uniforme, sin preferencia")
    print("  - Suavizado de Laplace: Evita probabilidades cero")
    print("\nUsos:")
    print("  - Punto de partida para inferencia bayesiana")
    print("  - Clasificación (Naive Bayes)")
    print("  - Predicción sin evidencia")
    print("  - Regularización en aprendizaje automático")

