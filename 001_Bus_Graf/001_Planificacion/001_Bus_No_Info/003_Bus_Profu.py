"""
Algoritmo 3: Búsqueda en Profundidad (DFS - Depth-First Search)

La búsqueda en profundidad explora tan profundo como sea posible a lo largo de cada rama
antes de retroceder. Utiliza una pila LIFO (Last In, First Out) o recursión.

Características:
- Completo: No es completo en grafos infinitos o con ciclos
- Óptimo: No garantiza encontrar la solución óptima
- Complejidad temporal: O(b^m) donde m es la profundidad máxima
- Complejidad espacial: O(bm) - mejor que BFS
"""

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


def busqueda_profundidad_iterativa(grafo: Grafo, inicio: str, objetivo: str) -> Optional[Tuple[List[str], int]]:
    """
    Realiza una búsqueda en profundidad iterativa desde el nodo inicio hasta el nodo objetivo.
    
    Args:
        grafo: El grafo donde realizar la búsqueda
        inicio: Nodo inicial
        objetivo: Nodo objetivo
    
    Returns:
        Una tupla con el camino encontrado y el número de nodos explorados,
        o None si no existe camino
    """
    # Pila para los nodos por explorar: (nodo, camino_hasta_nodo)
    pila = [(inicio, [inicio])]
    
    # Conjunto de nodos visitados
    visitados: Set[str] = set()
    
    # Contador de nodos explorados
    nodos_explorados = 0
    
    while pila:
        # Extraer el último nodo de la pila
        nodo_actual, camino = pila.pop()
        
        # Si ya fue visitado, continuar
        if nodo_actual in visitados:
            continue
        
        # Marcar como visitado
        visitados.add(nodo_actual)
        nodos_explorados += 1
        
        # Verificar si alcanzamos el objetivo
        if nodo_actual == objetivo:
            return camino, nodos_explorados
        
        # Explorar vecinos (en orden inverso para mantener el orden original)
        vecinos = grafo.obtener_vecinos(nodo_actual)
        for vecino in reversed(vecinos):
            if vecino not in visitados:
                nuevo_camino = camino + [vecino]
                pila.append((vecino, nuevo_camino))
    
    # No se encontró camino
    return None


def busqueda_profundidad_recursiva(grafo: Grafo, inicio: str, objetivo: str) -> Optional[Tuple[List[str], int]]:
    """
    Realiza una búsqueda en profundidad recursiva desde el nodo inicio hasta el nodo objetivo.
    
    Args:
        grafo: El grafo donde realizar la búsqueda
        inicio: Nodo inicial
        objetivo: Nodo objetivo
    
    Returns:
        Una tupla con el camino encontrado y el número de nodos explorados,
        o None si no existe camino
    """
    visitados: Set[str] = set()
    nodos_explorados = [0]  # Lista para poder modificar en la recursión
    
    def dfs_recursivo(nodo: str, camino: List[str]) -> Optional[List[str]]:
        """Función auxiliar recursiva"""
        visitados.add(nodo)
        nodos_explorados[0] += 1
        
        if nodo == objetivo:
            return camino
        
        for vecino in grafo.obtener_vecinos(nodo):
            if vecino not in visitados:
                resultado = dfs_recursivo(vecino, camino + [vecino])
                if resultado is not None:
                    return resultado
        
        return None
    
    camino = dfs_recursivo(inicio, [inicio])
    if camino:
        return camino, nodos_explorados[0]
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
    
    print("=== Búsqueda en Profundidad (DFS) - Versión Iterativa ===\n")
    
    # Realizar búsqueda iterativa
    inicio = "A"
    objetivo = "H"
    resultado = busqueda_profundidad_iterativa(g, inicio, objetivo)
    
    if resultado:
        camino, nodos_explorados = resultado
        print(f"Camino encontrado de {inicio} a {objetivo}:")
        print(" -> ".join(camino))
        print(f"\nNodos explorados: {nodos_explorados}")
        print(f"Longitud del camino: {len(camino)}")
    else:
        print(f"No se encontró camino de {inicio} a {objetivo}")
    
    print("\n" + "="*50 + "\n")
    print("=== Búsqueda en Profundidad (DFS) - Versión Recursiva ===\n")
    
    # Realizar búsqueda recursiva
    resultado2 = busqueda_profundidad_recursiva(g, inicio, objetivo)
    
    if resultado2:
        camino2, nodos_explorados2 = resultado2
        print(f"Camino encontrado de {inicio} a {objetivo}:")
        print(" -> ".join(camino2))
        print(f"\nNodos explorados: {nodos_explorados2}")
        print(f"Longitud del camino: {len(camino2)}")
    else:
        print(f"No se encontró camino de {inicio} a {objetivo}")
    
    # Comparación
    print("\n" + "="*50 + "\n")
    print("Nota: DFS puede encontrar diferentes caminos dependiendo del orden")
    print("de exploración de los vecinos. No garantiza el camino más corto.")

