"""
Algoritmo 10: Búsquedas A* y AO*

A* es el algoritmo de búsqueda informada más popular. Combina el costo real g(n)
con la estimación heurística h(n) mediante f(n) = g(n) + h(n).

AO* es una variante para grafos AND-OR, útil en problemas de descomposición.

Características de A*:
- Completo: Siempre encuentra una solución si existe
- Óptimo: Encuentra la solución óptima si h(n) es admisible
- Complejidad: Depende de la calidad de la heurística
"""

import heapq
import math
from typing import List, Dict, Optional, Set, Tuple, Callable


class NodoAEstrella:
    """Representa un nodo en la búsqueda A*"""
    
    def __init__(self, estado: Tuple[int, int], padre: Optional['NodoAEstrella'] = None,
                 costo_g: float = 0, heuristica_h: float = 0):
        self.estado = estado
        self.padre = padre
        self.g = costo_g  # Costo real desde el inicio
        self.h = heuristica_h  # Estimación heurística al objetivo
        self.f = self.g + self.h  # Función de evaluación
    
    def __lt__(self, otro):
        """Comparación basada en f(n) = g(n) + h(n)"""
        if self.f == otro.f:
            # Desempate por h(n) (preferir nodos más cerca del objetivo)
            return self.h < otro.h
        return self.f < otro.f
    
    def obtener_camino(self) -> List[Tuple[int, int]]:
        """Reconstruye el camino desde el inicio"""
        camino = []
        nodo = self
        while nodo is not None:
            camino.append(nodo.estado)
            nodo = nodo.padre
        camino.reverse()
        return camino


class Cuadricula:
    """Representa una cuadrícula para navegación"""
    
    def __init__(self, ancho: int, alto: int, obstaculos: Set[Tuple[int, int]] = None):
        self.ancho = ancho
        self.alto = alto
        self.obstaculos = obstaculos if obstaculos else set()
    
    def es_valido(self, pos: Tuple[int, int]) -> bool:
        """Verifica si una posición es válida"""
        x, y = pos
        return (0 <= x < self.ancho and 
                0 <= y < self.alto and 
                pos not in self.obstaculos)
    
    def obtener_vecinos(self, pos: Tuple[int, int], diagonal: bool = False) -> List[Tuple[Tuple[int, int], float]]:
        """Retorna los vecinos válidos de una posición con sus costos"""
        x, y = pos
        vecinos = []
        
        # Movimientos en 4 direcciones
        direcciones = [
            ((x+1, y), 1.0),   # Derecha
            ((x-1, y), 1.0),   # Izquierda
            ((x, y+1), 1.0),   # Abajo
            ((x, y-1), 1.0),   # Arriba
        ]
        
        # Movimientos diagonales
        if diagonal:
            direcciones.extend([
                ((x+1, y+1), math.sqrt(2)),  # Diagonal abajo-derecha
                ((x+1, y-1), math.sqrt(2)),  # Diagonal arriba-derecha
                ((x-1, y+1), math.sqrt(2)),  # Diagonal abajo-izquierda
                ((x-1, y-1), math.sqrt(2)),  # Diagonal arriba-izquierda
            ])
        
        for nueva_pos, costo in direcciones:
            if self.es_valido(nueva_pos):
                vecinos.append((nueva_pos, costo))
        
        return vecinos


def heuristica_manhattan(estado: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
    """Distancia de Manhattan (admisible para movimientos en 4 direcciones)"""
    return abs(estado[0] - objetivo[0]) + abs(estado[1] - objetivo[1])


def heuristica_euclidiana(estado: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
    """Distancia Euclidiana (admisible para cualquier movimiento)"""
    dx = estado[0] - objetivo[0]
    dy = estado[1] - objetivo[1]
    return math.sqrt(dx * dx + dy * dy)


def heuristica_octil(estado: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
    """Distancia Octil (admisible para movimientos en 8 direcciones)"""
    dx = abs(estado[0] - objetivo[0])
    dy = abs(estado[1] - objetivo[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


def a_estrella(cuadricula: Cuadricula,
               inicio: Tuple[int, int],
               objetivo: Tuple[int, int],
               heuristica: Callable = heuristica_manhattan,
               diagonal: bool = False) -> Optional[Tuple[List[Tuple[int, int]], float, int]]:
    """
    Implementación del algoritmo A*.
    
    Args:
        cuadricula: La cuadrícula donde buscar
        inicio: Posición inicial
        objetivo: Posición objetivo
        heuristica: Función heurística a usar
        diagonal: Permitir movimientos diagonales
    
    Returns:
        Una tupla con el camino, costo total y nodos explorados, o None si no hay solución
    """
    # Crear nodo inicial
    h_inicial = heuristica(inicio, objetivo)
    nodo_inicial = NodoAEstrella(inicio, None, 0, h_inicial)
    
    # Cola de prioridad (ordenada por f(n) = g(n) + h(n))
    frontera = [nodo_inicial]
    heapq.heapify(frontera)
    
    # Conjunto de estados explorados
    explorados: Set[Tuple[int, int]] = set()
    
    # Diccionario para rastrear el mejor costo g conocido a cada estado
    mejor_g: Dict[Tuple[int, int], float] = {inicio: 0}
    
    # Contador de nodos explorados
    nodos_explorados = 0
    
    while frontera:
        # Extraer nodo con menor f(n)
        nodo_actual = heapq.heappop(frontera)
        
        # Si ya fue explorado con un mejor costo, continuar
        if nodo_actual.estado in explorados:
            continue
        
        # Marcar como explorado
        explorados.add(nodo_actual.estado)
        nodos_explorados += 1
        
        # Verificar si alcanzamos el objetivo
        if nodo_actual.estado == objetivo:
            return nodo_actual.obtener_camino(), nodo_actual.g, nodos_explorados
        
        # Expandir vecinos
        for vecino_pos, costo_movimiento in cuadricula.obtener_vecinos(nodo_actual.estado, diagonal):
            if vecino_pos not in explorados:
                nuevo_g = nodo_actual.g + costo_movimiento
                
                # Solo agregar si encontramos un mejor camino o es nuevo
                if vecino_pos not in mejor_g or nuevo_g < mejor_g[vecino_pos]:
                    mejor_g[vecino_pos] = nuevo_g
                    h = heuristica(vecino_pos, objetivo)
                    nodo_vecino = NodoAEstrella(vecino_pos, nodo_actual, nuevo_g, h)
                    heapq.heappush(frontera, nodo_vecino)
    
    # No se encontró camino
    return None


def visualizar_camino(cuadricula: Cuadricula, camino: List[Tuple[int, int]],
                     inicio: Tuple[int, int], objetivo: Tuple[int, int]):
    """Visualiza el camino en la cuadrícula"""
    for y in range(cuadricula.alto):
        fila = ""
        for x in range(cuadricula.ancho):
            pos = (x, y)
            if pos == inicio:
                fila += "I "
            elif pos == objetivo:
                fila += "O "
            elif pos in cuadricula.obstaculos:
                fila += "█ "
            elif pos in camino:
                fila += "· "
            else:
                fila += "  "
        print(fila)


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Algoritmo A* (A Estrella) ===\n")
    
    # Crear cuadrícula con obstáculos
    obstaculos = {
        (2, 1), (2, 2), (2, 3), (2, 4),
        (5, 2), (5, 3), (5, 4), (5, 5),
        (7, 1), (7, 2), (7, 3)
    }
    cuadricula = Cuadricula(10, 8, obstaculos)
    
    inicio = (0, 0)
    objetivo = (9, 7)
    
    print(f"Inicio: {inicio}")
    print(f"Objetivo: {objetivo}\n")
    
    # Probar A* con movimientos en 4 direcciones
    print("--- A* con Movimientos en 4 Direcciones (Manhattan) ---\n")
    resultado = a_estrella(cuadricula, inicio, objetivo, heuristica_manhattan, diagonal=False)
    
    if resultado:
        camino, costo, nodos_explorados = resultado
        print(f"✓ Camino encontrado")
        print(f"Longitud del camino: {len(camino)}")
        print(f"Costo total: {costo:.2f}")
        print(f"Nodos explorados: {nodos_explorados}\n")
        
        print("Visualización (I=Inicio, O=Objetivo, █=Obstáculo, ·=Camino):\n")
        visualizar_camino(cuadricula, camino, inicio, objetivo)
    else:
        print("✗ No se encontró camino")
    
    print("\n" + "="*50 + "\n")
    
    # Probar A* con movimientos en 8 direcciones
    print("--- A* con Movimientos en 8 Direcciones (Octil) ---\n")
    resultado2 = a_estrella(cuadricula, inicio, objetivo, heuristica_octil, diagonal=True)
    
    if resultado2:
        camino2, costo2, nodos_explorados2 = resultado2
        print(f"✓ Camino encontrado")
        print(f"Longitud del camino: {len(camino2)}")
        print(f"Costo total: {costo2:.2f}")
        print(f"Nodos explorados: {nodos_explorados2}\n")
        
        print("Visualización:\n")
        visualizar_camino(cuadricula, camino2, inicio, objetivo)
    
    print("\n" + "="*50 + "\n")
    print("Propiedades de A*:")
    print("- Óptimo: Si h(n) es admisible (nunca sobreestima)")
    print("- Completo: Siempre encuentra solución si existe")
    print("- Eficiente: Expande menos nodos que búsqueda no informada")
    print("- f(n) = g(n) + h(n): Combina costo real y estimación")
    print("\nComparación:")
    if resultado and resultado2:
        print(f"4 direcciones: {resultado[2]} nodos, costo {resultado[1]:.2f}")
        print(f"8 direcciones: {resultado2[2]} nodos, costo {resultado2[1]:.2f}")

