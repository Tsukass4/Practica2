"""
Algoritmo 5: Búsqueda en Profundidad Iterativa (IDS - Iterative Deepening Search)

La búsqueda en profundidad iterativa combina las ventajas de BFS y DFS.
Realiza múltiples búsquedas en profundidad limitada con límites incrementales.

Características:
- Completo: Siempre encuentra una solución si existe
- Óptimo: Encuentra la solución con menor número de pasos
- Complejidad temporal: O(b^d) similar a BFS
- Complejidad espacial: O(bd) mejor que BFS
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


def dls_para_ids(grafo: Grafo, nodo: str, objetivo: str, limite: int, 
                 camino: List[str], visitados: Set[str], stats: Dict) -> Optional[List[str]]:
    """
    Función auxiliar de DLS para IDS.
    
    Args:
        grafo: El grafo donde realizar la búsqueda
        nodo: Nodo actual
        objetivo: Nodo objetivo
        limite: Profundidad máxima actual
        camino: Camino actual
        visitados: Conjunto de nodos visitados en el camino actual
        stats: Diccionario para estadísticas
    
    Returns:
        El camino si se encuentra el objetivo, None en caso contrario
    """
    stats['nodos_explorados'] += 1
    
    # Verificar si alcanzamos el objetivo
    if nodo == objetivo:
        return camino
    
    # Verificar si alcanzamos el límite de profundidad
    if limite <= 0:
        return None
    
    # Explorar vecinos
    for vecino in grafo.obtener_vecinos(nodo):
        if vecino not in visitados:
            nuevos_visitados = visitados.copy()
            nuevos_visitados.add(vecino)
            resultado = dls_para_ids(grafo, vecino, objetivo, limite - 1, 
                                     camino + [vecino], nuevos_visitados, stats)
            if resultado is not None:
                return resultado
    
    return None


def busqueda_profundidad_iterativa(grafo: Grafo, inicio: str, objetivo: str, 
                                   max_profundidad: int = 100) -> Optional[Tuple[List[str], int, int]]:
    """
    Realiza una búsqueda en profundidad iterativa desde el nodo inicio hasta el nodo objetivo.
    
    Args:
        grafo: El grafo donde realizar la búsqueda
        inicio: Nodo inicial
        objetivo: Nodo objetivo
        max_profundidad: Profundidad máxima a explorar (para evitar bucles infinitos)
    
    Returns:
        Una tupla con el camino encontrado, la profundidad en que se encontró y el número
        total de nodos explorados, o None si no existe camino
    """
    # Iterar incrementando el límite de profundidad
    for profundidad in range(max_profundidad):
        stats = {'nodos_explorados': 0}
        
        # Realizar DLS con el límite actual
        resultado = dls_para_ids(grafo, inicio, objetivo, profundidad, 
                                [inicio], {inicio}, stats)
        
        if resultado is not None:
            return resultado, profundidad, stats['nodos_explorados']
    
    # No se encontró camino dentro del límite máximo
    return None


def busqueda_profundidad_iterativa_verbose(grafo: Grafo, inicio: str, objetivo: str, 
                                          max_profundidad: int = 100) -> Optional[Tuple[List[str], int, int, List[int]]]:
    """
    Versión verbose de IDS que retorna información detallada de cada iteración.
    
    Args:
        grafo: El grafo donde realizar la búsqueda
        inicio: Nodo inicial
        objetivo: Nodo objetivo
        max_profundidad: Profundidad máxima a explorar
    
    Returns:
        Una tupla con el camino, profundidad final, nodos totales explorados y lista de
        nodos explorados por iteración, o None si no existe camino
    """
    nodos_por_iteracion = []
    nodos_totales = 0
    
    # Iterar incrementando el límite de profundidad
    for profundidad in range(max_profundidad):
        stats = {'nodos_explorados': 0}
        
        # Realizar DLS con el límite actual
        resultado = dls_para_ids(grafo, inicio, objetivo, profundidad, 
                                [inicio], {inicio}, stats)
        
        nodos_por_iteracion.append(stats['nodos_explorados'])
        nodos_totales += stats['nodos_explorados']
        
        if resultado is not None:
            return resultado, profundidad, nodos_totales, nodos_por_iteracion
    
    # No se encontró camino dentro del límite máximo
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
    
    print("=== Búsqueda en Profundidad Iterativa (IDS) ===\n")
    
    inicio = "A"
    objetivo = "H"
    
    # Realizar búsqueda
    resultado = busqueda_profundidad_iterativa(g, inicio, objetivo)
    
    if resultado:
        camino, profundidad, nodos_explorados = resultado
        print(f"Camino encontrado de {inicio} a {objetivo}:")
        print(" -> ".join(camino))
        print(f"\nProfundidad de la solución: {profundidad}")
        print(f"Nodos explorados totales: {nodos_explorados}")
        print(f"Longitud del camino: {len(camino)}")
    else:
        print(f"No se encontró camino de {inicio} a {objetivo}")
    
    print("\n" + "="*50 + "\n")
    print("=== Versión Detallada (Verbose) ===\n")
    
    # Realizar búsqueda verbose
    resultado2 = busqueda_profundidad_iterativa_verbose(g, inicio, objetivo)
    
    if resultado2:
        camino2, profundidad2, nodos_totales, nodos_por_iter = resultado2
        print(f"Camino encontrado de {inicio} a {objetivo}:")
        print(" -> ".join(camino2))
        print(f"\nProfundidad de la solución: {profundidad2}")
        print(f"Nodos explorados totales: {nodos_totales}")
        print(f"\nNodos explorados por iteración:")
        for i, nodos in enumerate(nodos_por_iter):
            print(f"  Profundidad {i}: {nodos} nodos")
    
    print("\n" + "="*50 + "\n")
    print("Ventajas de IDS:")
    print("- Combina la completitud y optimalidad de BFS")
    print("- Con el uso eficiente de memoria de DFS")
    print("- Aunque re-explora nodos, el overhead es aceptable")

