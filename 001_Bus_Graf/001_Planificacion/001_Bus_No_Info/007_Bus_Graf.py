"""
Algoritmo 7: Búsqueda en Grafos (General Graph Search)

Implementación general de búsqueda en grafos que evita estados repetidos.
Puede configurarse para usar diferentes estrategias (BFS, DFS, UCS, etc.).

Características:
- Evita ciclos mediante seguimiento de estados visitados
- Configurable con diferentes estrategias de frontera
- Base para implementar múltiples algoritmos de búsqueda
"""

from collections import deque
import heapq
from typing import List, Dict, Optional, Set, Tuple, Callable
from enum import Enum


class EstrategiaBusqueda(Enum):
    """Estrategias de búsqueda disponibles"""
    BFS = "breadth_first"
    DFS = "depth_first"
    UCS = "uniform_cost"


class Nodo:
    """Representa un nodo en el espacio de búsqueda"""
    
    def __init__(self, estado: str, padre: Optional['Nodo'] = None, 
                 accion: Optional[str] = None, costo: float = 0):
        self.estado = estado
        self.padre = padre
        self.accion = accion
        self.costo = costo
        self.profundidad = 0 if padre is None else padre.profundidad + 1
    
    def __lt__(self, otro):
        """Comparación para la cola de prioridad"""
        return self.costo < otro.costo
    
    def obtener_camino(self) -> List[str]:
        """Reconstruye el camino desde el inicio hasta este nodo"""
        camino = []
        nodo = self
        while nodo is not None:
            camino.append(nodo.estado)
            nodo = nodo.padre
        camino.reverse()
        return camino
    
    def obtener_acciones(self) -> List[str]:
        """Reconstruye las acciones desde el inicio hasta este nodo"""
        acciones = []
        nodo = self
        while nodo.padre is not None:
            if nodo.accion:
                acciones.append(nodo.accion)
            nodo = nodo.padre
        acciones.reverse()
        return acciones


class Problema:
    """Define un problema de búsqueda"""
    
    def __init__(self, estado_inicial: str, estado_objetivo: str, grafo: Dict[str, List[Tuple[str, float]]]):
        self.estado_inicial = estado_inicial
        self.estado_objetivo = estado_objetivo
        self.grafo = grafo
    
    def es_objetivo(self, estado: str) -> bool:
        """Verifica si un estado es el objetivo"""
        return estado == self.estado_objetivo
    
    def obtener_sucesores(self, estado: str) -> List[Tuple[str, str, float]]:
        """
        Retorna los sucesores de un estado.
        
        Returns:
            Lista de tuplas (estado_sucesor, accion, costo)
        """
        sucesores = []
        for vecino, costo in self.grafo.get(estado, []):
            accion = f"{estado}->{vecino}"
            sucesores.append((vecino, accion, costo))
        return sucesores


class Frontera:
    """Maneja la frontera de búsqueda con diferentes estrategias"""
    
    def __init__(self, estrategia: EstrategiaBusqueda):
        self.estrategia = estrategia
        if estrategia == EstrategiaBusqueda.BFS:
            self.contenedor = deque()
        elif estrategia == EstrategiaBusqueda.DFS:
            self.contenedor = []
        elif estrategia == EstrategiaBusqueda.UCS:
            self.contenedor = []
    
    def agregar(self, nodo: Nodo):
        """Agrega un nodo a la frontera"""
        if self.estrategia == EstrategiaBusqueda.BFS:
            self.contenedor.append(nodo)
        elif self.estrategia == EstrategiaBusqueda.DFS:
            self.contenedor.append(nodo)
        elif self.estrategia == EstrategiaBusqueda.UCS:
            heapq.heappush(self.contenedor, nodo)
    
    def extraer(self) -> Nodo:
        """Extrae un nodo de la frontera según la estrategia"""
        if self.estrategia == EstrategiaBusqueda.BFS:
            return self.contenedor.popleft()
        elif self.estrategia == EstrategiaBusqueda.DFS:
            return self.contenedor.pop()
        elif self.estrategia == EstrategiaBusqueda.UCS:
            return heapq.heappop(self.contenedor)
    
    def esta_vacia(self) -> bool:
        """Verifica si la frontera está vacía"""
        return len(self.contenedor) == 0


def busqueda_grafo(problema: Problema, estrategia: EstrategiaBusqueda) -> Optional[Tuple[Nodo, int]]:
    """
    Algoritmo general de búsqueda en grafos.
    
    Args:
        problema: El problema a resolver
        estrategia: La estrategia de búsqueda a usar
    
    Returns:
        Una tupla con el nodo solución y el número de nodos explorados,
        o None si no existe solución
    """
    # Crear nodo inicial
    nodo_inicial = Nodo(problema.estado_inicial)
    
    # Verificar si el estado inicial es el objetivo
    if problema.es_objetivo(nodo_inicial.estado):
        return nodo_inicial, 0
    
    # Inicializar frontera y explorados
    frontera = Frontera(estrategia)
    frontera.agregar(nodo_inicial)
    explorados: Set[str] = set()
    
    # Contador de nodos explorados
    nodos_explorados = 0
    
    # Búsqueda
    while not frontera.esta_vacia():
        # Extraer nodo de la frontera
        nodo = frontera.extraer()
        
        # Si ya fue explorado, continuar
        if nodo.estado in explorados:
            continue
        
        # Marcar como explorado
        explorados.add(nodo.estado)
        nodos_explorados += 1
        
        # Verificar si es el objetivo
        if problema.es_objetivo(nodo.estado):
            return nodo, nodos_explorados
        
        # Expandir nodo
        for estado_sucesor, accion, costo in problema.obtener_sucesores(nodo.estado):
            if estado_sucesor not in explorados:
                nodo_hijo = Nodo(
                    estado_sucesor,
                    padre=nodo,
                    accion=accion,
                    costo=nodo.costo + costo
                )
                frontera.agregar(nodo_hijo)
    
    # No se encontró solución
    return None


# Ejemplo de uso
if __name__ == "__main__":
    # Crear un grafo de ejemplo
    grafo = {
        "A": [("B", 4), ("C", 2)],
        "B": [("D", 5), ("E", 10)],
        "C": [("B", 1), ("F", 8)],
        "D": [("G", 2)],
        "E": [("G", 3)],
        "F": [("E", 2), ("G", 6)],
        "G": []
    }
    
    # Crear problema
    problema = Problema("A", "G", grafo)
    
    print("=== Búsqueda en Grafos - Comparación de Estrategias ===\n")
    
    # Probar diferentes estrategias
    estrategias = [
        (EstrategiaBusqueda.BFS, "Búsqueda en Anchura (BFS)"),
        (EstrategiaBusqueda.DFS, "Búsqueda en Profundidad (DFS)"),
        (EstrategiaBusqueda.UCS, "Búsqueda de Costo Uniforme (UCS)")
    ]
    
    for estrategia, nombre in estrategias:
        print(f"--- {nombre} ---")
        resultado = busqueda_grafo(problema, estrategia)
        
        if resultado:
            nodo_solucion, nodos_explorados = resultado
            camino = nodo_solucion.obtener_camino()
            acciones = nodo_solucion.obtener_acciones()
            
            print(f"Camino: {' -> '.join(camino)}")
            print(f"Costo total: {nodo_solucion.costo}")
            print(f"Profundidad: {nodo_solucion.profundidad}")
            print(f"Nodos explorados: {nodos_explorados}")
        else:
            print("No se encontró solución")
        
        print()
    
    print("="*50)
    print("\nObservaciones:")
    print("- BFS encuentra el camino con menos pasos")
    print("- UCS encuentra el camino de menor costo")
    print("- DFS puede encontrar soluciones rápido pero no garantiza optimalidad")

