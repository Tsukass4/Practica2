"""
Algoritmo 1: Búsqueda en Anchura (BFS - Breadth-First Search)

La búsqueda en anchura es un algoritmo de búsqueda no informada que explora
todos los nodos a un nivel de profundidad antes de pasar al siguiente nivel.
Utiliza una cola FIFO (First In, First Out) para mantener los nodos por explorar.

Características:
- Completo: Siempre encuentra una solución si existe
- Óptimo: Encuentra la solución con menor número de pasos
- Complejidad temporal: O(b^d) donde b es el factor de ramificación y d la profundidad
- Complejidad espacial: O(b^d)
"""

from collections import deque
from typing import List, Dict, Optional, Set, Tuple


class Grafo:
    """Representa un grafo para realizar búsquedas"""
    
    def __init__(self):
        self.adyacencias: Dict[str, List[str]] = {}
    
    def agregar_arista(self, origen: str, destino: str):
        """Agrega una arista al grafo"""
        if origen not in self.adyacencias:
            self.adyacencias[origen] = []
        if destino not in self.adyacencias:
            self.adyacencias[destino] = []
        self.adyacencias[origen].append(destino)
    
    def obtener_vecinos(self, nodo: str) -> List[str]:
        """Retorna los vecinos de un nodo"""
        return self.adyacencias.get(nodo, [])


def busqueda_anchura(grafo: Grafo, inicio: str, objetivo: str) -> Optional[Tuple[List[str], int]]:
    """
    Realiza una búsqueda en anchura desde el nodo inicio hasta el nodo objetivo.
    
    Args:
        grafo: El grafo donde realizar la búsqueda
        inicio: Nodo inicial
        objetivo: Nodo objetivo
    
    Returns:
        Una tupla con el camino encontrado y el número de nodos explorados,
        o None si no existe camino
    """
    # Cola para los nodos por explorar
    cola = deque([inicio])
    
    # Conjunto de nodos visitados
    visitados: Set[str] = {inicio}
    
    # Diccionario para reconstruir el camino
    padres: Dict[str, Optional[str]] = {inicio: None}
    
    # Contador de nodos explorados
    nodos_explorados = 0
    
    while cola:
        # Extraer el primer nodo de la cola
        nodo_actual = cola.popleft()
        nodos_explorados += 1
        
        # Verificar si alcanzamos el objetivo
        if nodo_actual == objetivo:
            # Reconstruir el camino
            camino = []
            nodo = objetivo
            while nodo is not None:
                camino.append(nodo)
                nodo = padres[nodo]
            camino.reverse()
            return camino, nodos_explorados
        
        # Explorar vecinos
        for vecino in grafo.obtener_vecinos(nodo_actual):
            if vecino not in visitados:
                visitados.add(vecino)
                padres[vecino] = nodo_actual
                cola.append(vecino)
    
    # No se encontró camino
    return None


# Ejemplo de uso
if __name__ == "__main__":
    # Crear un grafo de ejemplo
    g = Grafo()
    
    # Agregar aristas (grafo dirigido)
    g.agregar_arista("A", "B")
    g.agregar_arista("A", "C")
    g.agregar_arista("B", "D")
    g.agregar_arista("B", "E")
    g.agregar_arista("C", "F")
    g.agregar_arista("C", "G")
    g.agregar_arista("D", "H")
    g.agregar_arista("E", "H")
    g.agregar_arista("F", "H")
    
    print("=== Búsqueda en Anchura (BFS) ===\n")
    
    # Realizar búsqueda
    inicio = "A"
    objetivo = "H"
    resultado = busqueda_anchura(g, inicio, objetivo)
    
    if resultado:
        camino, nodos_explorados = resultado
        print(f"Camino encontrado de {inicio} a {objetivo}:")
        print(" -> ".join(camino))
        print(f"\nNodos explorados: {nodos_explorados}")
        print(f"Longitud del camino: {len(camino)}")
    else:
        print(f"No se encontró camino de {inicio} a {objetivo}")
    
    # Otro ejemplo
    print("\n" + "="*50 + "\n")
    objetivo2 = "G"
    resultado2 = busqueda_anchura(g, inicio, objetivo2)
    
    if resultado2:
        camino2, nodos_explorados2 = resultado2
        print(f"Camino encontrado de {inicio} a {objetivo2}:")
        print(" -> ".join(camino2))
        print(f"\nNodos explorados: {nodos_explorados2}")
        print(f"Longitud del camino: {len(camino2)}")

