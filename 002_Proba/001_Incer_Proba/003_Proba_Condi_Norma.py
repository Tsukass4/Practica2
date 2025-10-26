"""
Algoritmo 40: Probabilidad Condicionada y Normalización

La probabilidad condicionada P(A|B) es la probabilidad de A dado que B ha ocurrido.
La normalización asegura que las probabilidades sumen 1.

Fórmulas:
- P(A|B) = P(A ∩ B) / P(B)
- P(A|B) = P(B|A) * P(A) / P(B)  (Regla de Bayes)
- Normalización: α * [p1, p2, ...] donde α = 1/(p1 + p2 + ...)
"""

from typing import Dict, List, Tuple
from collections import defaultdict


class ProbabilidadCondicionada:
    """Manejo de probabilidades condicionadas"""
    
    def __init__(self):
        # P(A, B) - probabilidades conjuntas
        self.prob_conjunta = defaultdict(float)
        # P(A) - probabilidades marginales
        self.prob_marginal = defaultdict(float)
    
    def establecer_conjunta(self, evento_a: str, evento_b: str, probabilidad: float):
        """Establece P(A, B)"""
        self.prob_conjunta[(evento_a, evento_b)] = probabilidad
    
    def calcular_marginales(self, eventos_a: List[str], eventos_b: List[str]):
        """Calcula probabilidades marginales desde conjuntas"""
        # P(A) = Σ_b P(A, b)
        for a in eventos_a:
            self.prob_marginal[a] = sum(
                self.prob_conjunta[(a, b)] for b in eventos_b
            )
        
        # P(B) = Σ_a P(a, B)
        for b in eventos_b:
            self.prob_marginal[b] = sum(
                self.prob_conjunta[(a, b)] for a in eventos_a
            )
    
    def prob_condicionada(self, evento_a: str, dado_b: str) -> float:
        """
        Calcula P(A|B) = P(A, B) / P(B)
        """
        prob_b = self.prob_marginal.get(dado_b, 0)
        if prob_b == 0:
            return 0.0
        
        prob_a_y_b = self.prob_conjunta.get((evento_a, dado_b), 0)
        return prob_a_y_b / prob_b
    
    @staticmethod
    def normalizar(distribución: Dict[str, float]) -> Dict[str, float]:
        """Normaliza una distribución de probabilidad"""
        total = sum(distribución.values())
        if total == 0:
            return distribución
        
        return {evento: prob/total for evento, prob in distribución.items()}


# Ejemplo 1: Diagnóstico médico
def ejemplo_diagnostico_medico():
    """Ejemplo de probabilidad condicionada en diagnóstico"""
    print("=== Probabilidad Condicionada: Diagnóstico Médico ===\n")
    
    modelo = ProbabilidadCondicionada()
    
    # Tabla de probabilidad conjunta P(Enfermedad, Síntoma)
    # Enfermedad: {gripe, resfriado}
    # Síntoma: {fiebre, sin_fiebre}
    
    print("Tabla de Probabilidad Conjunta P(Enfermedad, Síntoma):")
    print("                  Fiebre    Sin Fiebre")
    print("  Gripe           0.08      0.02")
    print("  Resfriado       0.05      0.85")
    print()
    
    modelo.establecer_conjunta("gripe", "fiebre", 0.08)
    modelo.establecer_conjunta("gripe", "sin_fiebre", 0.02)
    modelo.establecer_conjunta("resfriado", "fiebre", 0.05)
    modelo.establecer_conjunta("resfriado", "sin_fiebre", 0.85)
    
    # Calcular marginales
    enfermedades = ["gripe", "resfriado"]
    sintomas = ["fiebre", "sin_fiebre"]
    modelo.calcular_marginales(enfermedades, sintomas)
    
    print("Probabilidades Marginales:")
    print(f"  P(Gripe) = {modelo.prob_marginal['gripe']:.2f}")
    print(f"  P(Resfriado) = {modelo.prob_marginal['resfriado']:.2f}")
    print(f"  P(Fiebre) = {modelo.prob_marginal['fiebre']:.2f}")
    print(f"  P(Sin Fiebre) = {modelo.prob_marginal['sin_fiebre']:.2f}")
    
    # Probabilidades condicionadas
    print("\nProbabilidades Condicionadas:")
    
    # P(Gripe | Fiebre)
    p_gripe_dado_fiebre = modelo.prob_condicionada("gripe", "fiebre")
    print(f"  P(Gripe | Fiebre) = {p_gripe_dado_fiebre:.3f}")
    
    # P(Resfriado | Fiebre)
    p_resfriado_dado_fiebre = modelo.prob_condicionada("resfriado", "fiebre")
    print(f"  P(Resfriado | Fiebre) = {p_resfriado_dado_fiebre:.3f}")
    
    # P(Fiebre | Gripe)
    p_fiebre_dado_gripe = modelo.prob_condicionada("fiebre", "gripe")
    print(f"  P(Fiebre | Gripe) = {p_fiebre_dado_gripe:.3f}")
    
    # P(Fiebre | Resfriado)
    p_fiebre_dado_resfriado = modelo.prob_condicionada("fiebre", "resfriado")
    print(f"  P(Fiebre | Resfriado) = {p_fiebre_dado_resfriado:.3f}")
    
    print("\nInterpretación:")
    print(f"  - Si el paciente tiene fiebre, hay {p_gripe_dado_fiebre*100:.1f}% de probabilidad de gripe")
    print(f"  - La fiebre es más común en gripe ({p_fiebre_dado_gripe*100:.0f}%) que en resfriado ({p_fiebre_dado_resfriado*100:.0f}%)")


# Ejemplo 2: Normalización
def ejemplo_normalizacion():
    """Ejemplo de normalización de distribuciones"""
    print("\n\n=== Normalización de Distribuciones ===\n")
    
    # Distribución no normalizada (resultado de cálculos)
    dist_no_norm = {
        "A": 0.12,
        "B": 0.18,
        "C": 0.06,
        "D": 0.24
    }
    
    print("Distribución No Normalizada:")
    total = sum(dist_no_norm.values())
    for evento, prob in dist_no_norm.items():
        print(f"  {evento}: {prob:.2f}")
    print(f"  Suma: {total:.2f} (≠ 1.0)")
    
    # Normalizar
    dist_norm = ProbabilidadCondicionada.normalizar(dist_no_norm)
    
    print("\nDistribución Normalizada:")
    total_norm = sum(dist_norm.values())
    for evento, prob in dist_norm.items():
        print(f"  {evento}: {prob:.3f}")
    print(f"  Suma: {total_norm:.3f} (= 1.0)")
    
    print("\nFactor de normalización α:")
    alpha = 1.0 / total
    print(f"  α = 1/{total:.2f} = {alpha:.3f}")


# Ejemplo 3: Filtro de spam
def ejemplo_filtro_spam():
    """Ejemplo de probabilidad condicionada en filtro de spam"""
    print("\n\n=== Probabilidad Condicionada: Filtro de Spam ===\n")
    
    modelo = ProbabilidadCondicionada()
    
    # P(Clase, Palabra)
    # Clase: {spam, no_spam}
    # Palabra: {oferta, reunión}
    
    print("Probabilidades Conjuntas P(Clase, Palabra):")
    print("                 'oferta'  'reunión'")
    print("  Spam           0.25      0.05")
    print("  No Spam        0.05      0.65")
    print()
    
    modelo.establecer_conjunta("spam", "oferta", 0.25)
    modelo.establecer_conjunta("spam", "reunion", 0.05)
    modelo.establecer_conjunta("no_spam", "oferta", 0.05)
    modelo.establecer_conjunta("no_spam", "reunion", 0.65)
    
    clases = ["spam", "no_spam"]
    palabras = ["oferta", "reunion"]
    modelo.calcular_marginales(clases, palabras)
    
    # P(Spam | "oferta")
    p_spam_dado_oferta = modelo.prob_condicionada("spam", "oferta")
    print(f"P(Spam | 'oferta') = {p_spam_dado_oferta:.3f}")
    
    # P(Spam | "reunión")
    p_spam_dado_reunion = modelo.prob_condicionada("spam", "reunion")
    print(f"P(Spam | 'reunión') = {p_spam_dado_reunion:.3f}")
    
    print("\nInterpretación:")
    print(f"  - Email con 'oferta': {p_spam_dado_oferta*100:.1f}% probabilidad de spam")
    print(f"  - Email con 'reunión': {p_spam_dado_reunion*100:.1f}% probabilidad de spam")


# Ejemplo 4: Clima
def ejemplo_clima():
    """Ejemplo de probabilidad condicionada en predicción del clima"""
    print("\n\n=== Probabilidad Condicionada: Predicción del Clima ===\n")
    
    modelo = ProbabilidadCondicionada()
    
    # P(Clima_Hoy, Clima_Mañana)
    print("Probabilidades de transición del clima:")
    print("                    Mañana:")
    print("Hoy:              Sol    Nube   Lluvia")
    print("  Soleado        0.60    0.25    0.05")
    print("  Nublado        0.20    0.40    0.20")
    print("  Lluvioso       0.10    0.30    0.40")
    print()
    
    # Establecer probabilidades conjuntas (asumiendo P(Hoy) uniforme = 1/3)
    modelo.establecer_conjunta("sol_hoy", "sol_manana", 0.60 * 1/3)
    modelo.establecer_conjunta("sol_hoy", "nube_manana", 0.25 * 1/3)
    modelo.establecer_conjunta("sol_hoy", "lluvia_manana", 0.05 * 1/3)
    
    modelo.establecer_conjunta("nube_hoy", "sol_manana", 0.20 * 1/3)
    modelo.establecer_conjunta("nube_hoy", "nube_manana", 0.40 * 1/3)
    modelo.establecer_conjunta("nube_hoy", "lluvia_manana", 0.20 * 1/3)
    
    modelo.establecer_conjunta("lluvia_hoy", "sol_manana", 0.10 * 1/3)
    modelo.establecer_conjunta("lluvia_hoy", "nube_manana", 0.30 * 1/3)
    modelo.establecer_conjunta("lluvia_hoy", "lluvia_manana", 0.40 * 1/3)
    
    # Calcular P(Mañana | Hoy = Soleado)
    print("Si hoy está soleado, probabilidades para mañana:")
    
    # Crear distribución no normalizada
    dist = {
        "Sol": 0.60,
        "Nube": 0.25,
        "Lluvia": 0.05
    }
    
    # Normalizar (aunque ya suma ~0.9, no exactamente 1)
    dist_norm = ProbabilidadCondicionada.normalizar(dist)
    
    for clima, prob in dist_norm.items():
        print(f"  P({clima} mañana | Sol hoy) = {prob:.3f}")


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Probabilidad Condicionada y Normalización ===\n")
    
    ejemplo_diagnostico_medico()
    ejemplo_normalizacion()
    ejemplo_filtro_spam()
    ejemplo_clima()
    
    print("\n" + "="*70)
    print("\nConceptos Clave:")
    print("  - P(A|B): Probabilidad de A dado B")
    print("  - P(A|B) = P(A,B) / P(B)")
    print("  - Normalización: Asegurar que Σ P = 1")
    print("  - α = 1 / Σ (factor de normalización)")
    print("\nAplicaciones:")
    print("  - Diagnóstico médico")
    print("  - Filtrado de spam")
    print("  - Predicción del clima")
    print("  - Inferencia bayesiana")

