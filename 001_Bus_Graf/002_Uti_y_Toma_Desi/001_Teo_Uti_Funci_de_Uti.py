"""
Algoritmo 24: Teoría de la Utilidad - Función de Utilidad

La teoría de la utilidad proporciona un marco para tomar decisiones racionales
bajo incertidumbre. La función de utilidad mapea estados o resultados a valores
numéricos que representan las preferencias del agente.

Conceptos clave:
- Utilidad esperada: E[U] = Σ P(s) * U(s)
- Axiomas de utilidad (Von Neumann-Morgenstern)
- Aversión al riesgo vs. propensión al riesgo
- Equivalente de certeza
"""

import math
from typing import List, Tuple, Callable, Dict
import random


class FuncionUtilidad:
    """Clase base para funciones de utilidad"""
    
    def calcular(self, valor: float) -> float:
        """Calcula la utilidad de un valor"""
        raise NotImplementedError
    
    def nombre(self) -> str:
        """Retorna el nombre de la función"""
        raise NotImplementedError


class UtilidadLineal(FuncionUtilidad):
    """Función de utilidad lineal: U(x) = x (neutral al riesgo)"""
    
    def calcular(self, valor: float) -> float:
        return valor
    
    def nombre(self) -> str:
        return "Lineal (Neutral al riesgo)"


class UtilidadLogaritmica(FuncionUtilidad):
    """Función de utilidad logarítmica: U(x) = log(x) (aversión al riesgo)"""
    
    def __init__(self, base: float = math.e):
        self.base = base
    
    def calcular(self, valor: float) -> float:
        if valor <= 0:
            return float('-inf')
        return math.log(valor, self.base)
    
    def nombre(self) -> str:
        return "Logarítmica (Aversión al riesgo)"


class UtilidadExponencial(FuncionUtilidad):
    """Función de utilidad exponencial: U(x) = 1 - e^(-x/R) donde R es aversión al riesgo"""
    
    def __init__(self, coef_aversion: float = 1.0):
        self.R = coef_aversion
    
    def calcular(self, valor: float) -> float:
        return 1 - math.exp(-valor / self.R)
    
    def nombre(self) -> str:
        return f"Exponencial (R={self.R})"


class UtilidadCuadratica(FuncionUtilidad):
    """Función de utilidad cuadrática: U(x) = x - (b/2)x²"""
    
    def __init__(self, b: float = 0.01):
        self.b = b
    
    def calcular(self, valor: float) -> float:
        return valor - (self.b / 2) * valor ** 2
    
    def nombre(self) -> str:
        return f"Cuadrática (b={self.b})"


class UtilidadPotencia(FuncionUtilidad):
    """Función de utilidad de potencia: U(x) = x^α"""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    
    def calcular(self, valor: float) -> float:
        if valor < 0:
            return -abs(valor) ** self.alpha
        return valor ** self.alpha
    
    def nombre(self) -> str:
        if self.alpha < 1:
            tipo = "Aversión al riesgo"
        elif self.alpha > 1:
            tipo = "Propensión al riesgo"
        else:
            tipo = "Neutral"
        return f"Potencia (α={self.alpha}, {tipo})"


class Loteria:
    """Representa una lotería (distribución de probabilidad sobre resultados)"""
    
    def __init__(self, resultados: List[Tuple[float, float]]):
        """
        Args:
            resultados: Lista de tuplas (valor, probabilidad)
        """
        self.resultados = resultados
        # Normalizar probabilidades
        suma_prob = sum(p for _, p in resultados)
        self.resultados = [(v, p/suma_prob) for v, p in resultados]
    
    def utilidad_esperada(self, funcion_utilidad: FuncionUtilidad) -> float:
        """Calcula la utilidad esperada de la lotería"""
        return sum(p * funcion_utilidad.calcular(v) for v, p in self.resultados)
    
    def valor_esperado(self) -> float:
        """Calcula el valor esperado (monetario) de la lotería"""
        return sum(p * v for v, p in self.resultados)
    
    def equivalente_certeza(self, funcion_utilidad: FuncionUtilidad, 
                           min_val: float = 0, max_val: float = 1000, 
                           precision: float = 0.01) -> float:
        """
        Calcula el equivalente de certeza: el valor cierto con la misma utilidad
        que la lotería.
        """
        utilidad_loteria = self.utilidad_esperada(funcion_utilidad)
        
        # Búsqueda binaria
        bajo, alto = min_val, max_val
        while alto - bajo > precision:
            medio = (bajo + alto) / 2
            utilidad_medio = funcion_utilidad.calcular(medio)
            
            if utilidad_medio < utilidad_loteria:
                bajo = medio
            else:
                alto = medio
        
        return (bajo + alto) / 2
    
    def prima_riesgo(self, funcion_utilidad: FuncionUtilidad) -> float:
        """
        Calcula la prima de riesgo: diferencia entre valor esperado y 
        equivalente de certeza.
        """
        ve = self.valor_esperado()
        ec = self.equivalente_certeza(funcion_utilidad, 0, ve * 2)
        return ve - ec


def comparar_decisiones(loterias: List[Tuple[str, Loteria]], 
                       funciones: List[FuncionUtilidad]):
    """Compara decisiones usando diferentes funciones de utilidad"""
    
    print("Comparación de Decisiones\n")
    print("="*70)
    
    for func in funciones:
        print(f"\n{func.nombre()}:")
        print("-" * 70)
        
        mejor_loteria = None
        mejor_utilidad = float('-inf')
        
        for nombre, loteria in loterias:
            ue = loteria.utilidad_esperada(func)
            ve = loteria.valor_esperado()
            ec = loteria.equivalente_certeza(func)
            pr = loteria.prima_riesgo(func)
            
            print(f"\n{nombre}:")
            print(f"  Valor Esperado: ${ve:.2f}")
            print(f"  Utilidad Esperada: {ue:.4f}")
            print(f"  Equivalente de Certeza: ${ec:.2f}")
            print(f"  Prima de Riesgo: ${pr:.2f}")
            
            if ue > mejor_utilidad:
                mejor_utilidad = ue
                mejor_loteria = nombre
        
        print(f"\n  → Mejor opción: {mejor_loteria}")


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Teoría de la Utilidad - Función de Utilidad ===\n")
    
    # Definir loterias (decisiones bajo incertidumbre)
    loterias = [
        ("Opción Segura", Loteria([(100, 1.0)])),
        ("Lotería 50-50", Loteria([(0, 0.5), (200, 0.5)])),
        ("Lotería Arriesgada", Loteria([(0, 0.7), (350, 0.3)])),
        ("Lotería Conservadora", Loteria([(50, 0.5), (150, 0.5)])),
    ]
    
    # Definir funciones de utilidad
    funciones = [
        UtilidadLineal(),
        UtilidadLogaritmica(),
        UtilidadExponencial(50),
        UtilidadPotencia(0.5),
        UtilidadPotencia(1.5),
    ]
    
    # Comparar decisiones
    comparar_decisiones(loterias, funciones)
    
    print("\n" + "="*70)
    print("\nObservaciones:")
    print("- Utilidad Lineal: Neutral al riesgo, elige por valor esperado")
    print("- Utilidad Logarítmica/Exponencial: Aversión al riesgo")
    print("- Utilidad Potencia (α<1): Aversión al riesgo")
    print("- Utilidad Potencia (α>1): Propensión al riesgo")
    print("\nConceptos clave:")
    print("- Equivalente de Certeza: Valor cierto con misma utilidad")
    print("- Prima de Riesgo: Cantidad que pagaría por evitar riesgo")
    print("- Utilidad Esperada: Criterio de decisión racional")

