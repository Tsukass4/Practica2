"""
Algoritmo 42: Independencia Condicional

Dos eventos A y B son condicionalmente independientes dado C si:
P(A, B | C) = P(A | C) * P(B | C)

Esto significa que conocer B no aporta información sobre A si ya conocemos C.

Notación: A ⊥ B | C

Importancia:
- Simplifica cálculos en redes bayesianas
- Reduce número de parámetros necesarios
- Fundamental para Naive Bayes
"""

from typing import Dict, List, Set
from collections import defaultdict
import math


class ModeloIndependencia:
    """Modelo para verificar independencia condicional"""
    
    def __init__(self):
        # P(A, B, C)
        self.prob_conjunta = {}
    
    def establecer_prob(self, a, b, c, probabilidad: float):
        """Establece P(A=a, B=b, C=c)"""
        self.prob_conjunta[(a, b, c)] = probabilidad
    
    def prob_marginal_c(self, c, valores_a: List, valores_b: List) -> float:
        """P(C=c) = Σ_a Σ_b P(a, b, c)"""
        return sum(
            self.prob_conjunta.get((a, b, c), 0)
            for a in valores_a
            for b in valores_b
        )
    
    def prob_condicionada_ab_dado_c(self, a, b, c, valores_a: List, valores_b: List) -> float:
        """P(A=a, B=b | C=c)"""
        prob_c = self.prob_marginal_c(c, valores_a, valores_b)
        if prob_c == 0:
            return 0
        return self.prob_conjunta.get((a, b, c), 0) / prob_c
    
    def prob_condicionada_a_dado_c(self, a, c, valores_a: List, valores_b: List) -> float:
        """P(A=a | C=c)"""
        prob_c = self.prob_marginal_c(c, valores_a, valores_b)
        if prob_c == 0:
            return 0
        suma = sum(self.prob_conjunta.get((a, b, c), 0) for b in valores_b)
        return suma / prob_c
    
    def prob_condicionada_b_dado_c(self, b, c, valores_a: List, valores_b: List) -> float:
        """P(B=b | C=c)"""
        prob_c = self.prob_marginal_c(c, valores_a, valores_b)
        if prob_c == 0:
            return 0
        suma = sum(self.prob_conjunta.get((a, b, c), 0) for a in valores_a)
        return suma / prob_c
    
    def verificar_independencia_condicional(self, valores_a: List, valores_b: List, 
                                           valores_c: List, tolerancia: float = 1e-5) -> bool:
        """
        Verifica si A ⊥ B | C
        Comprueba: P(A, B | C) = P(A | C) * P(B | C) para todos los valores
        """
        for c in valores_c:
            for a in valores_a:
                for b in valores_b:
                    # P(A, B | C)
                    p_ab_dado_c = self.prob_condicionada_ab_dado_c(a, b, c, valores_a, valores_b)
                    
                    # P(A | C) * P(B | C)
                    p_a_dado_c = self.prob_condicionada_a_dado_c(a, c, valores_a, valores_b)
                    p_b_dado_c = self.prob_condicionada_b_dado_c(b, c, valores_a, valores_b)
                    producto = p_a_dado_c * p_b_dado_c
                    
                    # Verificar igualdad
                    if not math.isclose(p_ab_dado_c, producto, abs_tol=tolerancia):
                        return False
        
        return True


# Ejemplo 1: Alarma de incendio
def ejemplo_alarma_incendio():
    """
    Ejemplo clásico de independencia condicional:
    - Fuego causa Humo y Alarma
    - Humo y Alarma son dependientes
    - Pero Humo ⊥ Alarma | Fuego (condicionalmente independientes dado Fuego)
    """
    print("=== Independencia Condicional: Alarma de Incendio ===\n")
    
    modelo = ModeloIndependencia()
    
    print("Escenario:")
    print("  - Fuego → Humo")
    print("  - Fuego → Alarma")
    print("  - ¿Humo ⊥ Alarma | Fuego?")
    print()
    
    # P(Fuego, Humo, Alarma)
    # Asumiendo independencia condicional
    
    # P(Fuego)
    p_fuego = 0.01
    p_no_fuego = 0.99
    
    # P(Humo | Fuego)
    p_humo_dado_fuego = 0.9
    p_humo_dado_no_fuego = 0.1
    
    # P(Alarma | Fuego)
    p_alarma_dado_fuego = 0.95
    p_alarma_dado_no_fuego = 0.05
    
    # Calcular conjuntas asumiendo independencia condicional
    # P(F, H, A) = P(F) * P(H|F) * P(A|F)
    
    modelo.establecer_prob("fuego", "humo", "alarma",
                          p_fuego * p_humo_dado_fuego * p_alarma_dado_fuego)
    modelo.establecer_prob("fuego", "humo", "no_alarma",
                          p_fuego * p_humo_dado_fuego * (1 - p_alarma_dado_fuego))
    modelo.establecer_prob("fuego", "no_humo", "alarma",
                          p_fuego * (1 - p_humo_dado_fuego) * p_alarma_dado_fuego)
    modelo.establecer_prob("fuego", "no_humo", "no_alarma",
                          p_fuego * (1 - p_humo_dado_fuego) * (1 - p_alarma_dado_fuego))
    
    modelo.establecer_prob("no_fuego", "humo", "alarma",
                          p_no_fuego * p_humo_dado_no_fuego * p_alarma_dado_no_fuego)
    modelo.establecer_prob("no_fuego", "humo", "no_alarma",
                          p_no_fuego * p_humo_dado_no_fuego * (1 - p_alarma_dado_no_fuego))
    modelo.establecer_prob("no_fuego", "no_humo", "alarma",
                          p_no_fuego * (1 - p_humo_dado_no_fuego) * p_alarma_dado_no_fuego)
    modelo.establecer_prob("no_fuego", "no_humo", "no_alarma",
                          p_no_fuego * (1 - p_humo_dado_no_fuego) * (1 - p_alarma_dado_no_fuego))
    
    # Verificar independencia condicional
    fuegos = ["fuego", "no_fuego"]
    humos = ["humo", "no_humo"]
    alarmas = ["alarma", "no_alarma"]
    
    es_independiente = modelo.verificar_independencia_condicional(humos, alarmas, fuegos)
    
    print(f"¿Humo ⊥ Alarma | Fuego? {es_independiente}")
    print("\nInterpretación:")
    if es_independiente:
        print("  ✓ Humo y Alarma son condicionalmente independientes dado Fuego")
        print("  - Si sabemos si hay fuego, conocer el humo no nos dice nada nuevo sobre la alarma")
        print("  - Ambos son causados por el fuego, pero no se causan entre sí")


# Ejemplo 2: Naive Bayes
def ejemplo_naive_bayes():
    """Ejemplo de independencia condicional en Naive Bayes"""
    print("\n\n=== Independencia Condicional: Naive Bayes ===\n")
    
    print("Clasificación de texto (Spam vs No Spam)")
    print("\nSupuesto de Naive Bayes:")
    print("  Las palabras son condicionalmente independientes dada la clase")
    print("  P(palabra1, palabra2 | clase) = P(palabra1 | clase) * P(palabra2 | clase)")
    print()
    
    # Ejemplo simplificado
    print("Ejemplo:")
    print("  Clase: Spam")
    print("  Palabras: 'oferta', 'gratis'")
    print()
    
    # Sin independencia condicional (tabla completa)
    print("Sin independencia condicional:")
    print("  Necesitamos: P('oferta', 'gratis' | Spam)")
    print("  Requiere: 2^n parámetros para n palabras")
    print()
    
    # Con independencia condicional
    print("Con independencia condicional (Naive Bayes):")
    print("  P('oferta', 'gratis' | Spam) = P('oferta' | Spam) * P('gratis' | Spam)")
    print("  Requiere: solo n parámetros")
    print()
    
    # Cálculo
    p_oferta_spam = 0.7
    p_gratis_spam = 0.6
    
    print(f"  P('oferta' | Spam) = {p_oferta_spam}")
    print(f"  P('gratis' | Spam) = {p_gratis_spam}")
    print(f"  P('oferta', 'gratis' | Spam) ≈ {p_oferta_spam * p_gratis_spam:.3f}")
    print()
    print("Ventaja: Reduce exponencialmente el número de parámetros")
    print("Desventaja: Asunción puede ser incorrecta (palabras correlacionadas)")


# Ejemplo 3: Red Bayesiana
def ejemplo_red_bayesiana():
    """Ejemplo de independencias en red bayesiana"""
    print("\n\n=== Independencia Condicional: Red Bayesiana ===\n")
    
    print("Red Bayesiana:")
    print()
    print("     Estación")
    print("       / \\")
    print("      /   \\")
    print("  Lluvia  Aspersores")
    print("      \\   /")
    print("       \\ /")
    print("     Césped Mojado")
    print()
    
    print("Independencias condicionales:")
    print("  1. Lluvia ⊥ Aspersores | Estación")
    print("     - Dado la estación, lluvia y aspersores son independientes")
    print()
    print("  2. Lluvia ⊥ Aspersores")
    print("     - Sin condicionar, NO son independientes (dependen de estación)")
    print()
    print("  3. Estación ⊥ Césped Mojado | {Lluvia, Aspersores}")
    print("     - Dado lluvia y aspersores, la estación no aporta info sobre césped")
    print()
    
    print("Beneficio:")
    print("  - Sin independencias: 2^4 = 16 parámetros")
    print("  - Con independencias: 2 + 2 + 2 + 4 = 10 parámetros")


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Independencia Condicional ===\n")
    
    ejemplo_alarma_incendio()
    ejemplo_naive_bayes()
    ejemplo_red_bayesiana()
    
    print("\n" + "="*70)
    print("\nConceptos Clave:")
    print("  - A ⊥ B | C: A y B independientes dado C")
    print("  - P(A, B | C) = P(A | C) * P(B | C)")
    print("  - Reduce complejidad computacional")
    print("\nAplicaciones:")
    print("  - Redes Bayesianas")
    print("  - Naive Bayes")
    print("  - Modelos gráficos probabilísticos")

