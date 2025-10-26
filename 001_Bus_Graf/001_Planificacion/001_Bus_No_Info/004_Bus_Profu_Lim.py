"""
Algoritmo 4: Búsqueda en Profundidad Limitada (DLS - Depth-Limited Search)

La búsqueda en profundidad limitada es una variante de DFS que impone un límite
en la profundidad de exploración para evitar búsquedas infinitas.

Características:
- Completo: No es completo si el límite es menor que la profundidad de la solución
- Óptimo: No garantiza encontrar la solución óptima
- Complejidad temporal: O(b^l) donde l es el límite de profundidad
- Complejidad espacial: O(bl)
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


def busqueda_profundidad_limitada(grafo: Grafo, inicio: str, objetivo: str, limite: int) -> Optional[Tuple[List[str], int, bool]]:
    """
    Realiza una búsqueda en profundidad limitada desde el nodo inicio hasta el nodo objetivo.
    
    Args:
        grafo: El grafo donde realizar la búsqueda
        inicio: Nodo inicial
        objetivo: Nodo objetivo
        limite: Profundidad máxima de búsqueda
    
    Returns:
        Una tupla con el camino encontrado, el número de nodos explorados y un booleano
        indicando si se alcanzó el límite, o None si no existe camino
    """
    nodos_explorados = [0]  # Lista para poder modificar en la recursión
    limite_alcanzado = [False]  # Indica si se alcanzó el límite
    
    def dls_recursivo(nodo: str, camino: List[str], profundidad: int, visitados: Set[str]) -> Optional[List[str]]:
        """Función auxiliar recursiva"""
        visitados.add(nodo)
        nodos_explorados[0] += 1
        
        # Verificar si alcanzamos el objetivo
        if nodo == objetivo:
            return camino
        
        # Verificar si alcanzamos el límite de profundidad
        if profundidad >= limite:
            limite_alcanzado[0] = True
            return None
        
        # Explorar vecinos
        for vecino in grafo.obtener_vecinos(nodo):
            if vecino not in visitados:
                # Crear una copia del conjunto de visitados para cada rama
                nuevos_visitados = visitados.copy()
                resultado = dls_recursivo(vecino, camino + [vecino], profundidad + 1, nuevos_visitados)
                if resultado is not None:
                    return resultado
        
        return None
    
    camino = dls_recursivo(inicio, [inicio], 0, set())
    if camino:
        return camino, nodos_explorados[0], limite_alcanzado[0]
    return None


def busqueda_profundidad_limitada_iterativa(grafo: Grafo, inicio: str, objetivo: str, limite: int) -> Optional[Tuple[List[str], int, bool]]:
    """
    Realiza una búsqueda en profundidad limitada iterativa desde el nodo inicio hasta el nodo objetivo.
    
    Args:
        grafo: El grafo donde realizar la búsqueda
        inicio: Nodo inicial
        objetivo: Nodo objetivo
        limite: Profundidad máxima de búsqueda
    
    Returns:
        Una tupla con el camino encontrado, el número de nodos explorados y un booleano
        indicando si se alcanzó el límite, o None si no existe camino
    """
    # Pila: (nodo, camino, profundidad, visitados_en_camino)
    pila = [(inicio, [inicio], 0, {inicio})]
    
    nodos_explorados = 0
    limite_alcanzado = False
    
    while pila:
        nodo_actual, camino, profundidad, visitados = pila.pop()
        nodos_explorados += 1
        
        # Verificar si alcanzamos el objetivo
        if nodo_actual == objetivo:
            return camino, nodos_explorados, limite_alcanzado
        
        # Verificar si alcanzamos el límite de profundidad
        if profundidad >= limite:
            limite_alcanzado = True
            continue
        
        # Explorar vecinos (en orden inverso para mantener el orden original)
        vecinos = grafo.obtener_vecinos(nodo_actual)
        for vecino in reversed(vecinos):
            if vecino not in visitados:
                nuevos_visitados = visitados.copy()
                nuevos_visitados.add(vecino)
                nuevo_camino = camino + [vecino]
                pila.append((vecino, nuevo_camino, profundidad + 1, nuevos_visitados))
    
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
    
    print("=== Búsqueda en Profundidad Limitada (DLS) ===\n")
    
    inicio = "A"
    objetivo = "H"
    
    # Probar con diferentes límites
    for limite in [1, 2, 3, 4]:
        print(f"Límite de profundidad: {limite}")
        resultado = busqueda_profundidad_limitada(g, inicio, objetivo, limite)
        
        if resultado:
            camino, nodos_explorados, limite_alcanzado = resultado
            print(f"  ✓ Camino encontrado: {' -> '.join(camino)}")
            print(f"  Nodos explorados: {nodos_explorados}")
            print(f"  Longitud del camino: {len(camino)}")
            if limite_alcanzado:
                print(f"  ⚠ Se alcanzó el límite de profundidad en algunas ramas")
        else:
            print(f"  ✗ No se encontró camino con este límite")
        print()
    
    print("="*50 + "\n")
    print("=== Versión Iterativa ===\n")
    
    # Probar versión iterativa
    limite = 3
    print(f"Límite de profundidad: {limite}")
    resultado2 = busqueda_profundidad_limitada_iterativa(g, inicio, objetivo, limite)
    
    if resultado2:
        camino2, nodos_explorados2, limite_alcanzado2 = resultado2
        print(f"  ✓ Camino encontrado: {' -> '.join(camino2)}")
        print(f"  Nodos explorados: {nodos_explorados2}")
        print(f"  Longitud del camino: {len(camino2)}")
        if limite_alcanzado2:
            print(f"  ⚠ Se alcanzó el límite de profundidad en algunas ramas")
    else:
        print(f"  ✗ No se encontró camino con este límite")

