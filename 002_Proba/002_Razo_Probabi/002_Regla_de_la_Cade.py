"""
Algoritmo 45: Regla de la Cadena

La regla de la cadena descompone una probabilidad conjunta en producto de condicionales:
P(X1, X2, ..., Xn) = P(X1) * P(X2|X1) * P(X3|X1,X2) * ... * P(Xn|X1,...,Xn-1)

En redes bayesianas se simplifica usando independencias:
P(X1, ..., Xn) = ∏_i P(Xi | Padres(Xi))
"""

from typing import Dict, List
from collections import defaultdict


class ReglaCadena:
    """Implementación de la regla de la cadena"""
    
    @staticmethod
    def prob_conjunta_general(probabilidades_condicionales: List[float]) -> float:
        """
        Calcula P(X1,...,Xn) = ∏ P(Xi | X1,...,Xi-1)
        
        Args:
            probabilidades_condicionales: Lista [P(X1), P(X2|X1), P(X3|X1,X2), ...]
        """
        resultado = 1.0
        for prob in probabilidades_condicionales:
            resultado *= prob
        return resultado
    
    @staticmethod
    def prob_conjunta_red_bayesiana(probs_dado_padres: List[float]) -> float:
        """
        Calcula P(X1,...,Xn) = ∏ P(Xi | Padres(Xi))
        
        Args:
            probs_dado_padres: Lista de P(Xi | Padres(Xi))
        """
        resultado = 1.0
        for prob in probs_dado_padres:
            resultado *= prob
        return resultado


# Ejemplo 1: Regla de la cadena general
def ejemplo_general():
    print("=== Regla de la Cadena: Forma General ===\n")
    
    print("Calcular: P(A, B, C)")
    print("\nRegla de la cadena:")
    print("  P(A, B, C) = P(A) * P(B|A) * P(C|A,B)")
    print()
    
    # Valores de ejemplo
    p_a = 0.6
    p_b_dado_a = 0.7
    p_c_dado_ab = 0.8
    
    print(f"  P(A) = {p_a}")
    print(f"  P(B|A) = {p_b_dado_a}")
    print(f"  P(C|A,B) = {p_c_dado_ab}")
    print()
    
    prob_conjunta = ReglaCadena.prob_conjunta_general([p_a, p_b_dado_a, p_c_dado_ab])
    
    print(f"  P(A, B, C) = {p_a} × {p_b_dado_a} × {p_c_dado_ab}")
    print(f"             = {prob_conjunta:.4f}")


# Ejemplo 2: Con red bayesiana
def ejemplo_red_bayesiana():
    print("\n\n=== Regla de la Cadena: Red Bayesiana ===\n")
    
    print("Red: A → B → C")
    print()
    print("Independencias:")
    print("  - B independiente de todo excepto A")
    print("  - C independiente de A dado B")
    print()
    print("Simplificación:")
    print("  P(A, B, C) = P(A) * P(B|A) * P(C|B)")
    print("  (en lugar de P(A) * P(B|A) * P(C|A,B))")
    print()
    
    p_a = 0.3
    p_b_dado_a = 0.8
    p_c_dado_b = 0.6
    
    print(f"  P(A) = {p_a}")
    print(f"  P(B|A) = {p_b_dado_a}")
    print(f"  P(C|B) = {p_c_dado_b}")
    print()
    
    prob = ReglaCadena.prob_conjunta_red_bayesiana([p_a, p_b_dado_a, p_c_dado_b])
    print(f"  P(A, B, C) = {prob:.4f}")


if __name__ == "__main__":
    print("=== Regla de la Cadena ===\n")
    ejemplo_general()
    ejemplo_red_bayesiana()
    
    print("\n" + "="*70)
    print("\nFormas de la Regla de la Cadena:")
    print("  General: P(X1,...,Xn) = ∏ P(Xi | X1,...,Xi-1)")
    print("  Red Bayesiana: P(X1,...,Xn) = ∏ P(Xi | Padres(Xi))")
    print("\nVentaja: Reduce complejidad usando independencias")

