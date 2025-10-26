"""
Algoritmo 8: Heurísticas

Las heurísticas son funciones que estiman el costo desde un estado hasta el objetivo.
Son fundamentales para los algoritmos de búsqueda informada.

Propiedades importantes:
- Admisibilidad: h(n) ≤ h*(n) (nunca sobreestima el costo real)
- Consistencia: h(n) ≤ c(n,a,n') + h(n') (desigualdad triangular)
"""

import math
from typing import Tuple, Dict, Callable
from abc import ABC, abstractmethod


class Heuristica(ABC):
    """Clase base abstracta para heurísticas"""
    
    @abstractmethod
    def calcular(self, estado: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
        """Calcula el valor heurístico desde el estado hasta el objetivo"""
        pass
    
    @abstractmethod
    def nombre(self) -> str:
        """Retorna el nombre de la heurística"""
        pass


class HeuristicaManhattan(Heuristica):
    """
    Distancia de Manhattan (L1)
    Suma de las diferencias absolutas en cada dimensión.
    Admisible para movimientos en 4 direcciones (arriba, abajo, izquierda, derecha).
    """
    
    def calcular(self, estado: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
        return abs(estado[0] - objetivo[0]) + abs(estado[1] - objetivo[1])
    
    def nombre(self) -> str:
        return "Manhattan"


class HeuristicaEuclidiana(Heuristica):
    """
    Distancia Euclidiana (L2)
    Distancia en línea recta entre dos puntos.
    Admisible para movimientos en cualquier dirección.
    """
    
    def calcular(self, estado: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
        dx = estado[0] - objetivo[0]
        dy = estado[1] - objetivo[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def nombre(self) -> str:
        return "Euclidiana"


class HeuristicaChebyshev(Heuristica):
    """
    Distancia de Chebyshev (L∞)
    Máximo de las diferencias absolutas en cada dimensión.
    Admisible para movimientos en 8 direcciones (incluyendo diagonales).
    """
    
    def calcular(self, estado: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
        return max(abs(estado[0] - objetivo[0]), abs(estado[1] - objetivo[1]))
    
    def nombre(self) -> str:
        return "Chebyshev"


class HeuristicaOctil(Heuristica):
    """
    Distancia Octil
    Combinación de movimientos diagonales y rectos.
    Más precisa que Chebyshev para movimientos en 8 direcciones.
    """
    
    def calcular(self, estado: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
        dx = abs(estado[0] - objetivo[0])
        dy = abs(estado[1] - objetivo[1])
        # sqrt(2) ≈ 1.414 es el costo de movimiento diagonal
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)
    
    def nombre(self) -> str:
        return "Octil"


class HeuristicaDiagonal(Heuristica):
    """
    Distancia Diagonal
    Similar a Octil pero con costo diagonal = 1.
    """
    
    def calcular(self, estado: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
        dx = abs(estado[0] - objetivo[0])
        dy = abs(estado[1] - objetivo[1])
        return max(dx, dy)
    
    def nombre(self) -> str:
        return "Diagonal"


class HeuristicaPersonalizada:
    """
    Permite crear heurísticas personalizadas mediante funciones
    """
    
    def __init__(self, funcion: Callable, nombre: str, admisible: bool = True):
        self.funcion = funcion
        self._nombre = nombre
        self.admisible = admisible
    
    def calcular(self, estado: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
        return self.funcion(estado, objetivo)
    
    def nombre(self) -> str:
        return self._nombre


def verificar_admisibilidad(heuristica: Heuristica, 
                           estados: list, 
                           objetivo: Tuple[int, int],
                           costos_reales: Dict[Tuple[int, int], float]) -> bool:
    """
    Verifica si una heurística es admisible comparando con costos reales conocidos.
    
    Args:
        heuristica: La heurística a verificar
        estados: Lista de estados a verificar
        objetivo: El estado objetivo
        costos_reales: Diccionario con los costos reales desde cada estado al objetivo
    
    Returns:
        True si la heurística es admisible, False en caso contrario
    """
    admisible = True
    
    for estado in estados:
        h_valor = heuristica.calcular(estado, objetivo)
        costo_real = costos_reales.get(estado, float('inf'))
        
        if h_valor > costo_real:
            print(f"  ✗ No admisible en {estado}: h={h_valor:.2f} > costo_real={costo_real:.2f}")
            admisible = False
    
    return admisible


def verificar_consistencia(heuristica: Heuristica,
                          transiciones: list) -> bool:
    """
    Verifica si una heurística es consistente.
    
    Args:
        heuristica: La heurística a verificar
        transiciones: Lista de tuplas (estado1, estado2, costo, objetivo)
    
    Returns:
        True si la heurística es consistente, False en caso contrario
    """
    consistente = True
    
    for estado1, estado2, costo, objetivo in transiciones:
        h1 = heuristica.calcular(estado1, objetivo)
        h2 = heuristica.calcular(estado2, objetivo)
        
        # h(n) ≤ c(n,a,n') + h(n')
        if h1 > costo + h2:
            print(f"  ✗ No consistente: h({estado1})={h1:.2f} > {costo} + h({estado2})={h2:.2f}")
            consistente = False
    
    return consistente


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Heurísticas para Búsqueda Informada ===\n")
    
    # Definir estados de ejemplo en una cuadrícula 2D
    inicio = (0, 0)
    objetivo = (5, 5)
    estados_intermedios = [(1, 1), (2, 3), (3, 2), (4, 4)]
    
    # Crear instancias de heurísticas
    heuristicas = [
        HeuristicaManhattan(),
        HeuristicaEuclidiana(),
        HeuristicaChebyshev(),
        HeuristicaOctil(),
        HeuristicaDiagonal()
    ]
    
    print("Comparación de heurísticas:")
    print(f"Inicio: {inicio}, Objetivo: {objetivo}\n")
    
    for h in heuristicas:
        valor = h.calcular(inicio, objetivo)
        print(f"{h.nombre():12s}: {valor:.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # Evaluar heurísticas en diferentes estados
    print("Valores heurísticos para diferentes estados:\n")
    print(f"{'Estado':<12} {'Manhattan':<12} {'Euclidiana':<12} {'Chebyshev':<12} {'Octil':<12}")
    print("-" * 60)
    
    for estado in [inicio] + estados_intermedios + [objetivo]:
        valores = [h.calcular(estado, objetivo) for h in heuristicas[:4]]
        print(f"{str(estado):<12} {valores[0]:<12.2f} {valores[1]:<12.2f} {valores[2]:<12.2f} {valores[3]:<12.2f}")
    
    print("\n" + "="*50 + "\n")
    
    # Verificar admisibilidad (asumiendo movimientos en 4 direcciones)
    print("Verificación de admisibilidad (movimientos en 4 direcciones):\n")
    
    # Costos reales conocidos (Manhattan es óptima para 4 direcciones)
    costos_reales = {
        (0, 0): 10,
        (1, 1): 8,
        (2, 3): 5,
        (3, 2): 5,
        (4, 4): 2,
        (5, 5): 0
    }
    
    h_manhattan = HeuristicaManhattan()
    h_euclidiana = HeuristicaEuclidiana()
    
    print("Manhattan:")
    if verificar_admisibilidad(h_manhattan, list(costos_reales.keys()), objetivo, costos_reales):
        print("  ✓ Admisible")
    
    print("\nEuclidiana:")
    if verificar_admisibilidad(h_euclidiana, list(costos_reales.keys()), objetivo, costos_reales):
        print("  ✓ Admisible")
    
    print("\n" + "="*50 + "\n")
    print("Propiedades de las heurísticas:")
    print("- Manhattan: Admisible para movimientos en 4 direcciones")
    print("- Euclidiana: Admisible para movimientos en cualquier dirección")
    print("- Chebyshev: Admisible para movimientos en 8 direcciones")
    print("- Una heurística consistente es siempre admisible")

