"""
Algoritmo 9: Búsqueda Voraz Primero el Mejor (Greedy Best-First Search)

La búsqueda voraz expande el nodo que parece estar más cerca del objetivo
según la función heurística h(n). Es "voraz" porque siempre elige la opción
que parece mejor localmente.

Características:
- Completo: No es completo (puede quedar atrapado en bucles)
- Óptimo: No garantiza encontrar la solución óptima
- Complejidad: O(b^m) en el peor caso
- Usa solo h(n) para evaluar nodos
"""

import heapq
import math
from typing import List, Dict, Optional, Set, Tuple, Callable


class Nodo:
    """Representa un nodo en el espacio de búsqueda"""
    
    def __init__(self, estado: Tuple[int, int], padre: Optional['Nodo'] = None, 
                 costo_g: float = 0, heuristica_h: float = 0):
        self.estado = estado
        self.padre = padre
        self.g = costo_g  # Costo desde el inicio
        self.h = heuristica_h  # Estimación heurística al objetivo
    
    def __lt__(self, otro):
        """Comparación basada solo en h(n) para búsqueda voraz"""
        return self.h < otro.h
    
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
    
    def obtener_vecinos(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
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
        
        for nueva_pos, costo in direcciones:
            if self.es_valido(nueva_pos):
                vecinos.append((nueva_pos, costo))
        
        return vecinos


def heuristica_manhattan(estado: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
    """Calcula la distancia de Manhattan"""
    return abs(estado[0] - objetivo[0]) + abs(estado[1] - objetivo[1])


def heuristica_euclidiana(estado: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
    """Calcula la distancia Euclidiana"""
    dx = estado[0] - objetivo[0]
    dy = estado[1] - objetivo[1]
    return math.sqrt(dx * dx + dy * dy)


def busqueda_voraz(cuadricula: Cuadricula, 
                   inicio: Tuple[int, int], 
                   objetivo: Tuple[int, int],
                   heuristica: Callable = heuristica_manhattan) -> Optional[Tuple[List[Tuple[int, int]], float, int]]:
    """
    Realiza una búsqueda voraz primero el mejor.
    
    Args:
        cuadricula: La cuadrícula donde buscar
        inicio: Posición inicial
        objetivo: Posición objetivo
        heuristica: Función heurística a usar
    
    Returns:
        Una tupla con el camino, costo total y nodos explorados, o None si no hay solución
    """
    # Crear nodo inicial
    h_inicial = heuristica(inicio, objetivo)
    nodo_inicial = Nodo(inicio, None, 0, h_inicial)
    
    # Cola de prioridad (ordenada por h(n))
    frontera = [nodo_inicial]
    heapq.heapify(frontera)
    
    # Conjunto de estados explorados
    explorados: Set[Tuple[int, int]] = set()
    
    # Diccionario para rastrear el mejor costo conocido a cada estado
    mejor_g: Dict[Tuple[int, int], float] = {inicio: 0}
    
    # Contador de nodos explorados
    nodos_explorados = 0
    
    while frontera:
        # Extraer nodo con menor h(n)
        nodo_actual = heapq.heappop(frontera)
        
        # Si ya fue explorado, continuar
        if nodo_actual.estado in explorados:
            continue
        
        # Marcar como explorado
        explorados.add(nodo_actual.estado)
        nodos_explorados += 1
        
        # Verificar si alcanzamos el objetivo
        if nodo_actual.estado == objetivo:
            return nodo_actual.obtener_camino(), nodo_actual.g, nodos_explorados
        
        # Expandir vecinos
        for vecino_pos, costo_movimiento in cuadricula.obtener_vecinos(nodo_actual.estado):
            if vecino_pos not in explorados:
                nuevo_g = nodo_actual.g + costo_movimiento
                
                # Solo agregar si encontramos un mejor camino o es nuevo
                if vecino_pos not in mejor_g or nuevo_g < mejor_g[vecino_pos]:
                    mejor_g[vecino_pos] = nuevo_g
                    h = heuristica(vecino_pos, objetivo)
                    nodo_vecino = Nodo(vecino_pos, nodo_actual, nuevo_g, h)
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
    print("=== Búsqueda Voraz Primero el Mejor ===\n")
    
    # Crear cuadrícula con obstáculos
    obstaculos = {
        (2, 1), (2, 2), (2, 3), (2, 4),
        (5, 2), (5, 3), (5, 4), (5, 5)
    }
    cuadricula = Cuadricula(10, 8, obstaculos)
    
    inicio = (0, 0)
    objetivo = (9, 7)
    
    print(f"Inicio: {inicio}")
    print(f"Objetivo: {objetivo}\n")
    
    # Probar con heurística Manhattan
    print("--- Usando Heurística Manhattan ---\n")
    resultado = busqueda_voraz(cuadricula, inicio, objetivo, heuristica_manhattan)
    
    if resultado:
        camino, costo, nodos_explorados = resultado
        print(f"Camino encontrado:")
        print(f"Longitud del camino: {len(camino)}")
        print(f"Costo total: {costo}")
        print(f"Nodos explorados: {nodos_explorados}\n")
        
        print("Visualización del camino (I=Inicio, O=Objetivo, █=Obstáculo, ·=Camino):\n")
        visualizar_camino(cuadricula, camino, inicio, objetivo)
    else:
        print("No se encontró camino")
    
    print("\n" + "="*50 + "\n")
    
    # Probar con heurística Euclidiana
    print("--- Usando Heurística Euclidiana ---\n")
    resultado2 = busqueda_voraz(cuadricula, inicio, objetivo, heuristica_euclidiana)
    
    if resultado2:
        camino2, costo2, nodos_explorados2 = resultado2
        print(f"Camino encontrado:")
        print(f"Longitud del camino: {len(camino2)}")
        print(f"Costo total: {costo2}")
        print(f"Nodos explorados: {nodos_explorados2}\n")
        
        print("Visualización del camino:\n")
        visualizar_camino(cuadricula, camino2, inicio, objetivo)
    
    print("\n" + "="*50 + "\n")
    print("Características de la Búsqueda Voraz:")
    print("- Rápida: Expande pocos nodos si la heurística es buena")
    print("- No óptima: Puede encontrar caminos subóptimos")
    print("- Sensible a la heurística: El resultado depende mucho de h(n)")

