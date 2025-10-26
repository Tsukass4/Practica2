"""
Algoritmo 23: Acondicionamiento del Corte (Cutset Conditioning)

El acondicionamiento del corte es una técnica para resolver CSPs
descomponiendo el problema. Se basa en encontrar un conjunto de
variables (cutset) cuya eliminación hace que el grafo de restricciones
sea acíclico (árbol).

Proceso:
1. Encontrar un cutset (conjunto de corte)
2. Asignar valores al cutset
3. Resolver el CSP restante (que es un árbol) eficientemente

Complejidad: O(d^c * (n-c)d²) donde c es el tamaño del cutset
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from collections import deque
import random


class CSP:
    """Clase CSP para acondicionamiento del corte"""
    
    def __init__(self, variables, dominios, restricciones):
        self.variables = variables
        self.dominios = dominios
        self.restricciones = restricciones
        self.vecinos = self._calcular_vecinos()
    
    def _calcular_vecinos(self):
        vecinos = {var: set() for var in self.variables}
        for variables_restriccion in self.restricciones.keys():
            for var1 in variables_restriccion:
                for var2 in variables_restriccion:
                    if var1 != var2:
                        vecinos[var1].add(var2)
        return vecinos
    
    def es_consistente(self, var, valor, asignacion):
        asignacion_temp = asignacion.copy()
        asignacion_temp[var] = valor
        
        for variables_restriccion, funcion_restriccion in self.restricciones.items():
            if var in variables_restriccion:
                if all(v in asignacion_temp for v in variables_restriccion):
                    valores = tuple(asignacion_temp[v] for v in variables_restriccion)
                    if not funcion_restriccion(*valores):
                        return False
        return True


def encontrar_cutset_simple(csp: CSP) -> Set[str]:
    """
    Encuentra un cutset simple usando heurística greedy.
    Selecciona variables con más vecinos hasta que el grafo sea acíclico.
    """
    cutset = set()
    variables_restantes = set(csp.variables)
    
    while not es_aciclico(csp, variables_restantes):
        # Seleccionar variable con más vecinos en el subgrafo
        var_max_grado = max(variables_restantes, 
                           key=lambda v: len(csp.vecinos[v] & variables_restantes))
        cutset.add(var_max_grado)
        variables_restantes.remove(var_max_grado)
    
    return cutset


def es_aciclico(csp: CSP, variables: Set[str]) -> bool:
    """Verifica si el subgrafo inducido por 'variables' es acíclico"""
    if len(variables) <= 1:
        return True
    
    # Usar BFS para detectar ciclos
    visitados = set()
    
    for inicio in variables:
        if inicio not in visitados:
            if tiene_ciclo_bfs(csp, inicio, variables, visitados):
                return False
    
    return True


def tiene_ciclo_bfs(csp: CSP, inicio: str, variables: Set[str], visitados: Set[str]) -> bool:
    """Detecta ciclos usando BFS"""
    cola = deque([(inicio, None)])
    visitados.add(inicio)
    
    while cola:
        nodo, padre = cola.popleft()
        
        for vecino in csp.vecinos[nodo]:
            if vecino not in variables:
                continue
            
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append((vecino, nodo))
            elif vecino != padre:
                return True  # Ciclo detectado
    
    return False


def resolver_arbol_csp(csp: CSP, variables: Set[str], asignacion: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Resuelve un CSP que es un árbol usando algoritmo dirigido.
    Complejidad: O(nd²)
    """
    if not variables:
        return asignacion
    
    # Seleccionar una raíz arbitraria
    raiz = next(iter(variables))
    
    # Ordenar variables topológicamente (de hojas a raíz)
    orden = ordenar_topologicamente(csp, raiz, variables)
    
    # Fase 1: Hacer arco-consistente de hojas a raíz
    for var in reversed(orden):
        if var in asignacion:
            continue
        
        # Eliminar valores inconsistentes
        valores_validos = []
        for valor in csp.dominios[var]:
            if csp.es_consistente(var, valor, asignacion):
                valores_validos.append(valor)
        
        if not valores_validos:
            return None  # Fallo
        
        csp.dominios[var] = valores_validos
    
    # Fase 2: Asignar valores de raíz a hojas
    for var in orden:
        if var not in asignacion:
            # Seleccionar primer valor consistente
            for valor in csp.dominios[var]:
                if csp.es_consistente(var, valor, asignacion):
                    asignacion[var] = valor
                    break
            else:
                return None  # No hay valor consistente
    
    return asignacion


def ordenar_topologicamente(csp: CSP, raiz: str, variables: Set[str]) -> List[str]:
    """Ordena variables topológicamente usando DFS desde la raíz"""
    visitados = set()
    orden = []
    
    def dfs(var):
        visitados.add(var)
        for vecino in csp.vecinos[var]:
            if vecino in variables and vecino not in visitados:
                dfs(vecino)
        orden.append(var)
    
    dfs(raiz)
    return orden


def acondicionamiento_cutset(csp: CSP) -> Optional[Dict[str, Any]]:
    """
    Resuelve CSP usando acondicionamiento del corte.
    
    Returns:
        Asignación completa si tiene éxito, None si falla
    """
    # Paso 1: Encontrar cutset
    cutset = encontrar_cutset_simple(csp)
    print(f"Cutset encontrado: {cutset} (tamaño: {len(cutset)})")
    
    # Paso 2: Generar todas las asignaciones posibles del cutset
    variables_cutset = list(cutset)
    dominios_cutset = [csp.dominios[var] for var in variables_cutset]
    
    # Probar cada combinación de valores para el cutset
    for asignacion_cutset in generar_combinaciones(variables_cutset, dominios_cutset):
        # Paso 3: Resolver el CSP restante (que es un árbol)
        variables_restantes = set(csp.variables) - cutset
        
        # Crear subproblema
        solucion = resolver_arbol_csp(csp, variables_restantes, asignacion_cutset.copy())
        
        if solucion is not None:
            return solucion
    
    return None


def generar_combinaciones(variables: List[str], dominios: List[List[Any]]) -> List[Dict[str, Any]]:
    """Genera todas las combinaciones de asignaciones para las variables"""
    if not variables:
        return [{}]
    
    combinaciones = []
    
    def backtrack(idx, asignacion_actual):
        if idx == len(variables):
            combinaciones.append(asignacion_actual.copy())
            return
        
        var = variables[idx]
        for valor in dominios[idx]:
            asignacion_actual[var] = valor
            backtrack(idx + 1, asignacion_actual)
            del asignacion_actual[var]
    
    backtrack(0, {})
    return combinaciones


class ProblemaReinas(CSP):
    """Problema de N-Reinas"""
    
    def __init__(self, n=8):
        self.n = n
        variables = [f"Q{i}" for i in range(n)]
        dominios = {var: list(range(n)) for var in variables}
        
        restricciones = {}
        for i in range(n):
            for j in range(i + 1, n):
                var1, var2 = f"Q{i}", f"Q{j}"
                restricciones[(var1, var2)] = lambda v1, v2, col1=i, col2=j: (
                    v1 != v2 and abs(v1 - v2) != abs(col1 - col2)
                )
        
        super().__init__(variables, dominios, restricciones)


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Acondicionamiento del Corte (Cutset Conditioning) ===\n")
    
    # Ejemplo con 4-Reinas (para demostración)
    print("--- 4-Reinas con Acondicionamiento del Corte ---\n")
    problema = ProblemaReinas(4)
    
    print("Estructura del problema:")
    print(f"Variables: {problema.variables}")
    print(f"Vecinos de Q0: {problema.vecinos['Q0']}")
    print(f"Vecinos de Q1: {problema.vecinos['Q1']}\n")
    
    # Encontrar cutset
    cutset = encontrar_cutset_simple(problema)
    print(f"Cutset: {cutset}")
    
    variables_restantes = set(problema.variables) - cutset
    print(f"Variables restantes: {variables_restantes}")
    print(f"¿Subgrafo restante es acíclico? {es_aciclico(problema, variables_restantes)}\n")
    
    # Resolver con acondicionamiento
    print("Resolviendo con acondicionamiento del corte...")
    solucion = acondicionamiento_cutset(problema)
    
    if solucion:
        print(f"\n✓ Solución encontrada: {solucion}\n")
        
        # Visualizar
        tablero = [['.' for _ in range(4)] for _ in range(4)]
        for i in range(4):
            fila = solucion[f"Q{i}"]
            tablero[fila][i] = 'Q'
        for fila in tablero:
            print(' '.join(fila))
    else:
        print("\n✗ No se encontró solución")
    
    print("\n" + "="*50)
    print("\nCaracterísticas del Acondicionamiento del Corte:")
    print("- Descompone el problema en cutset + árbol")
    print("- Cutset: Conjunto cuya eliminación hace el grafo acíclico")
    print("- Árbol CSP: Se resuelve eficientemente en O(nd²)")
    print("\nComplejidad:")
    print("- Total: O(d^c * (n-c)d²)")
    print("- c = tamaño del cutset")
    print("- d = tamaño del dominio")
    print("- n = número de variables")
    print("\nVentajas:")
    print("- Eficiente si el cutset es pequeño")
    print("- Garantiza encontrar solución si existe")
    print("\nDesventajas:")
    print("- Exponencial en el tamaño del cutset")
    print("- Encontrar el cutset mínimo es NP-hard")

