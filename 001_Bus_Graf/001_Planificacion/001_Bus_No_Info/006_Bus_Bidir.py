"""
Algoritmo 6: Búsqueda Bidireccional (Bidirectional Search)

La búsqueda bidireccional ejecuta dos búsquedas simultáneas: una desde el nodo inicial
hacia adelante y otra desde el nodo objetivo hacia atrás, hasta que se encuentran.

Características:
- Completo: Siempre encuentra una solución si existe
- Óptimo: Encuentra la solución óptima si ambas búsquedas son BFS
- Complejidad temporal: O(b^(d/2)) - mucho mejor que BFS
- Complejidad espacial: O(b^(d/2))
"""

from collections import deque
from typing import List, Dict, Optional, Set, Tuple


class Grafo:
    """Representa un grafo para realizar búsquedas"""
    
    def __init__(self):
        self.adyacencias: Dict[str, List[str]] = {}
        self.adyacencias_inversas: Dict[str, List[str]] = {}
    
    def agregar_arista(self, origen: str, destino: str):
        """Agrega una arista al grafo"""
        if origen not in self.adyacencias:
            self.adyacencias[origen] = []
        if destino not in self.adyacencias:
            self.adyacencias[destino] = []
        if origen not in self.adyacencias_inversas:
            self.adyacencias_inversas[origen] = []
        if destino not in self.adyacencias_inversas:
            self.adyacencias_inversas[destino] = []
        
        self.adyacencias[origen].append(destino)
        self.adyacencias_inversas[destino].append(origen)
    
    def obtener_vecinos(self, nodo: str) -> List[str]:
        """Retorna los vecinos hacia adelante de un nodo"""
        return self.adyacencias.get(nodo, [])
    
    def obtener_vecinos_inversos(self, nodo: str) -> List[str]:
        """Retorna los vecinos hacia atrás de un nodo"""
        return self.adyacencias_inversas.get(nodo, [])


def busqueda_bidireccional(grafo: Grafo, inicio: str, objetivo: str) -> Optional[Tuple[List[str], int]]:
    """
    Realiza una búsqueda bidireccional desde el nodo inicio hasta el nodo objetivo.
    
    Args:
        grafo: El grafo donde realizar la búsqueda
        inicio: Nodo inicial
        objetivo: Nodo objetivo
    
    Returns:
        Una tupla con el camino encontrado y el número de nodos explorados,
        o None si no existe camino
    """
    # Si inicio y objetivo son el mismo
    if inicio == objetivo:
        return [inicio], 0
    
    # Colas para ambas búsquedas
    cola_inicio = deque([inicio])
    cola_objetivo = deque([objetivo])
    
    # Conjuntos de visitados
    visitados_inicio: Set[str] = {inicio}
    visitados_objetivo: Set[str] = {objetivo}
    
    # Diccionarios de padres para reconstruir caminos
    padres_inicio: Dict[str, Optional[str]] = {inicio: None}
    padres_objetivo: Dict[str, Optional[str]] = {objetivo: None}
    
    # Contador de nodos explorados
    nodos_explorados = 0
    
    # Alternar entre búsquedas
    while cola_inicio and cola_objetivo:
        # Búsqueda desde el inicio
        nodo_punto_encuentro = expandir_nivel(
            grafo, cola_inicio, visitados_inicio, padres_inicio,
            visitados_objetivo, True
        )
        nodos_explorados += len(cola_inicio)
        
        if nodo_punto_encuentro:
            # Reconstruir camino completo
            camino = reconstruir_camino_bidireccional(
                nodo_punto_encuentro, padres_inicio, padres_objetivo
            )
            return camino, nodos_explorados
        
        # Búsqueda desde el objetivo
        nodo_punto_encuentro = expandir_nivel(
            grafo, cola_objetivo, visitados_objetivo, padres_objetivo,
            visitados_inicio, False
        )
        nodos_explorados += len(cola_objetivo)
        
        if nodo_punto_encuentro:
            # Reconstruir camino completo
            camino = reconstruir_camino_bidireccional(
                nodo_punto_encuentro, padres_inicio, padres_objetivo
            )
            return camino, nodos_explorados
    
    # No se encontró camino
    return None


def expandir_nivel(grafo: Grafo, cola: deque, visitados: Set[str], 
                   padres: Dict[str, Optional[str]], visitados_otro: Set[str],
                   hacia_adelante: bool) -> Optional[str]:
    """
    Expande un nivel de la búsqueda.
    
    Args:
        grafo: El grafo
        cola: Cola de nodos a explorar
        visitados: Conjunto de nodos visitados en esta dirección
        padres: Diccionario de padres en esta dirección
        visitados_otro: Conjunto de nodos visitados en la otra dirección
        hacia_adelante: True si es búsqueda hacia adelante, False si es hacia atrás
    
    Returns:
        El nodo de encuentro si se encontró, None en caso contrario
    """
    nivel_size = len(cola)
    
    for _ in range(nivel_size):
        nodo_actual = cola.popleft()
        
        # Obtener vecinos según la dirección
        if hacia_adelante:
            vecinos = grafo.obtener_vecinos(nodo_actual)
        else:
            vecinos = grafo.obtener_vecinos_inversos(nodo_actual)
        
        for vecino in vecinos:
            # Verificar si encontramos un nodo visitado por la otra búsqueda
            if vecino in visitados_otro:
                padres[vecino] = nodo_actual
                return vecino
            
            # Si no ha sido visitado, agregarlo
            if vecino not in visitados:
                visitados.add(vecino)
                padres[vecino] = nodo_actual
                cola.append(vecino)
    
    return None


def reconstruir_camino_bidireccional(punto_encuentro: str, 
                                     padres_inicio: Dict[str, Optional[str]],
                                     padres_objetivo: Dict[str, Optional[str]]) -> List[str]:
    """
    Reconstruye el camino completo desde el inicio hasta el objetivo.
    
    Args:
        punto_encuentro: Nodo donde se encontraron las búsquedas
        padres_inicio: Diccionario de padres desde el inicio
        padres_objetivo: Diccionario de padres desde el objetivo
    
    Returns:
        El camino completo
    """
    # Camino desde inicio hasta punto de encuentro
    camino_inicio = []
    nodo = punto_encuentro
    while nodo is not None:
        camino_inicio.append(nodo)
        nodo = padres_inicio[nodo]
    camino_inicio.reverse()
    
    # Camino desde punto de encuentro hasta objetivo
    camino_objetivo = []
    nodo = padres_objetivo[punto_encuentro]
    while nodo is not None:
        camino_objetivo.append(nodo)
        nodo = padres_objetivo[nodo]
    
    # Combinar ambos caminos
    return camino_inicio + camino_objetivo


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
    g.agregar_arista("F", "I")
    g.agregar_arista("G", "I")
    g.agregar_arista("H", "J")
    g.agregar_arista("I", "J")
    
    print("=== Búsqueda Bidireccional ===\n")
    
    inicio = "A"
    objetivo = "J"
    
    # Realizar búsqueda
    resultado = busqueda_bidireccional(g, inicio, objetivo)
    
    if resultado:
        camino, nodos_explorados = resultado
        print(f"Camino encontrado de {inicio} a {objetivo}:")
        print(" -> ".join(camino))
        print(f"\nNodos explorados: {nodos_explorados}")
        print(f"Longitud del camino: {len(camino)}")
    else:
        print(f"No se encontró camino de {inicio} a {objetivo}")
    
    print("\n" + "="*50 + "\n")
    print("Ventajas de la Búsqueda Bidireccional:")
    print("- Reduce significativamente el número de nodos explorados")
    print("- Complejidad O(b^(d/2)) en lugar de O(b^d)")
    print("- Especialmente útil cuando el factor de ramificación es alto")
    print("\nDesventajas:")
    print("- Requiere que el grafo sea reversible")
    print("- Necesita más memoria para mantener dos búsquedas")

