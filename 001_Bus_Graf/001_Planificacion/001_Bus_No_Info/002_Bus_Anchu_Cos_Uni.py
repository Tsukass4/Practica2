"""
Algoritmo 2: Búsqueda en Anchura de Costo Uniforme (UCS - Uniform Cost Search)

La búsqueda de costo uniforme expande el nodo con el menor costo de camino acumulado.
Utiliza una cola de prioridad para seleccionar siempre el nodo con menor costo.

Características:
- Completo: Siempre encuentra una solución si existe
- Óptimo: Encuentra la solución de menor costo
- Complejidad temporal: O(b^(1+⌊C*/ε⌋)) donde C* es el costo óptimo y ε el costo mínimo
- Complejidad espacial: O(b^(1+⌊C*/ε⌋))
"""

import heapq
from typing import List, Dict, Optional, Tuple


class GrafoPonderado:
    """Representa un grafo con pesos en las aristas"""
    
    def __init__(self):
        self.adyacencias: Dict[str, List[Tuple[str, float]]] = {}
    
    def agregar_arista(self, origen: str, destino: str, costo: float):
        """Agrega una arista con costo al grafo"""
        if origen not in self.adyacencias:
            self.adyacencias[origen] = []
        if destino not in self.adyacencias:
            self.adyacencias[destino] = []
        self.adyacencias[origen].append((destino, costo))
    
    def obtener_vecinos(self, nodo: str) -> List[Tuple[str, float]]:
        """Retorna los vecinos de un nodo con sus costos"""
        return self.adyacencias.get(nodo, [])


def busqueda_costo_uniforme(grafo: GrafoPonderado, inicio: str, objetivo: str) -> Optional[Tuple[List[str], float, int]]:
    """
    Realiza una búsqueda de costo uniforme desde el nodo inicio hasta el nodo objetivo.
    
    Args:
        grafo: El grafo ponderado donde realizar la búsqueda
        inicio: Nodo inicial
        objetivo: Nodo objetivo
    
    Returns:
        Una tupla con el camino encontrado, el costo total y el número de nodos explorados,
        o None si no existe camino
    """
    # Cola de prioridad: (costo_acumulado, nodo)
    cola_prioridad = [(0, inicio)]
    
    # Diccionario de costos mínimos conocidos
    costos: Dict[str, float] = {inicio: 0}
    
    # Diccionario para reconstruir el camino
    padres: Dict[str, Optional[str]] = {inicio: None}
    
    # Conjunto de nodos explorados
    explorados = set()
    
    # Contador de nodos explorados
    nodos_explorados = 0
    
    while cola_prioridad:
        # Extraer el nodo con menor costo acumulado
        costo_actual, nodo_actual = heapq.heappop(cola_prioridad)
        
        # Si ya fue explorado, continuar
        if nodo_actual in explorados:
            continue
        
        # Marcar como explorado
        explorados.add(nodo_actual)
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
            return camino, costo_actual, nodos_explorados
        
        # Explorar vecinos
        for vecino, costo_arista in grafo.obtener_vecinos(nodo_actual):
            nuevo_costo = costo_actual + costo_arista
            
            # Si encontramos un camino mejor o es la primera vez que visitamos el nodo
            if vecino not in costos or nuevo_costo < costos[vecino]:
                costos[vecino] = nuevo_costo
                padres[vecino] = nodo_actual
                heapq.heappush(cola_prioridad, (nuevo_costo, vecino))
    
    # No se encontró camino
    return None


# Ejemplo de uso
if __name__ == "__main__":
    # Crear un grafo ponderado de ejemplo
    g = GrafoPonderado()
    
    # Agregar aristas con costos
    g.agregar_arista("A", "B", 4)
    g.agregar_arista("A", "C", 2)
    g.agregar_arista("B", "D", 5)
    g.agregar_arista("B", "E", 10)
    g.agregar_arista("C", "B", 1)
    g.agregar_arista("C", "F", 8)
    g.agregar_arista("D", "G", 2)
    g.agregar_arista("E", "G", 3)
    g.agregar_arista("F", "E", 2)
    g.agregar_arista("F", "G", 6)
    
    print("=== Búsqueda de Costo Uniforme (UCS) ===\n")
    
    # Realizar búsqueda
    inicio = "A"
    objetivo = "G"
    resultado = busqueda_costo_uniforme(g, inicio, objetivo)
    
    if resultado:
        camino, costo_total, nodos_explorados = resultado
        print(f"Camino encontrado de {inicio} a {objetivo}:")
        print(" -> ".join(camino))
        print(f"\nCosto total: {costo_total}")
        print(f"Nodos explorados: {nodos_explorados}")
        print(f"Longitud del camino: {len(camino)}")
    else:
        print(f"No se encontró camino de {inicio} a {objetivo}")
    
    # Otro ejemplo
    print("\n" + "="*50 + "\n")
    objetivo2 = "E"
    resultado2 = busqueda_costo_uniforme(g, inicio, objetivo2)
    
    if resultado2:
        camino2, costo_total2, nodos_explorados2 = resultado2
        print(f"Camino encontrado de {inicio} a {objetivo2}:")
        print(" -> ".join(camino2))
        print(f"\nCosto total: {costo_total2}")
        print(f"Nodos explorados: {nodos_explorados2}")
        print(f"Longitud del camino: {len(camino2)}")

