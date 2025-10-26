"""
Algoritmo 17: Problemas de Satisfacción de Restricciones (CSP)

Un CSP está definido por:
- Variables: X = {X1, X2, ..., Xn}
- Dominios: D = {D1, D2, ..., Dn}
- Restricciones: C que especifican combinaciones permitidas de valores

Ejemplos clásicos:
- N-Reinas
- Coloración de grafos
- Sudoku
- Problema de asignación
"""

from typing import List, Dict, Set, Tuple, Callable, Optional, Any


class CSP:
    """Clase base para Problemas de Satisfacción de Restricciones"""
    
    def __init__(self, variables: List[str], dominios: Dict[str, List[Any]], 
                 restricciones: Dict[Tuple[str, ...], Callable]):
        """
        Args:
            variables: Lista de variables
            dominios: Diccionario variable -> lista de valores posibles
            restricciones: Diccionario tupla_variables -> función_restricción
        """
        self.variables = variables
        self.dominios = dominios
        self.restricciones = restricciones
        self.vecinos = self._calcular_vecinos()
    
    def _calcular_vecinos(self) -> Dict[str, Set[str]]:
        """Calcula los vecinos de cada variable (variables que comparten restricciones)"""
        vecinos = {var: set() for var in self.variables}
        
        for variables_restriccion in self.restricciones.keys():
            for var1 in variables_restriccion:
                for var2 in variables_restriccion:
                    if var1 != var2:
                        vecinos[var1].add(var2)
        
        return vecinos
    
    def es_consistente(self, var: str, valor: Any, asignacion: Dict[str, Any]) -> bool:
        """Verifica si asignar valor a var es consistente con la asignación actual"""
        asignacion_temp = asignacion.copy()
        asignacion_temp[var] = valor
        
        # Verificar todas las restricciones que involucran a var
        for variables_restriccion, funcion_restriccion in self.restricciones.items():
            if var in variables_restriccion:
                # Solo verificar si todas las variables de la restricción están asignadas
                if all(v in asignacion_temp for v in variables_restriccion):
                    valores = tuple(asignacion_temp[v] for v in variables_restriccion)
                    if not funcion_restriccion(*valores):
                        return False
        
        return True
    
    def obtener_valores_legales(self, var: str, asignacion: Dict[str, Any]) -> List[Any]:
        """Retorna los valores del dominio de var que son consistentes con la asignación"""
        return [valor for valor in self.dominios[var] 
                if self.es_consistente(var, valor, asignacion)]


class ProblemaReinas(CSP):
    """Problema de las N-Reinas como CSP"""
    
    def __init__(self, n: int = 8):
        self.n = n
        variables = [f"Q{i}" for i in range(n)]
        dominios = {var: list(range(n)) for var in variables}
        
        # Crear restricciones: cada par de reinas no debe atacarse
        restricciones = {}
        for i in range(n):
            for j in range(i + 1, n):
                var1, var2 = f"Q{i}", f"Q{j}"
                restricciones[(var1, var2)] = lambda v1, v2, col1=i, col2=j: (
                    v1 != v2 and  # No misma fila
                    abs(v1 - v2) != abs(col1 - col2)  # No misma diagonal
                )
        
        super().__init__(variables, dominios, restricciones)


class ProblemaColoracionGrafo(CSP):
    """Problema de coloración de grafos como CSP"""
    
    def __init__(self, nodos: List[str], aristas: List[Tuple[str, str]], num_colores: int = 3):
        variables = nodos
        colores = [f"Color{i}" for i in range(num_colores)]
        dominios = {nodo: colores.copy() for nodo in nodos}
        
        # Restricción: nodos adyacentes deben tener colores diferentes
        restricciones = {}
        for nodo1, nodo2 in aristas:
            restricciones[(nodo1, nodo2)] = lambda c1, c2: c1 != c2
        
        super().__init__(variables, dominios, restricciones)


class ProblemaSudoku(CSP):
    """Problema de Sudoku como CSP"""
    
    def __init__(self, tablero_inicial: List[List[int]]):
        """
        Args:
            tablero_inicial: Matriz 9x9 con valores iniciales (0 = vacío)
        """
        # Variables: cada celda
        variables = [f"C{i}{j}" for i in range(9) for j in range(9)]
        
        # Dominios
        dominios = {}
        for i in range(9):
            for j in range(9):
                var = f"C{i}{j}"
                if tablero_inicial[i][j] != 0:
                    dominios[var] = [tablero_inicial[i][j]]
                else:
                    dominios[var] = list(range(1, 10))
        
        # Restricciones
        restricciones = {}
        
        # Restricciones de fila
        for i in range(9):
            for j1 in range(9):
                for j2 in range(j1 + 1, 9):
                    var1, var2 = f"C{i}{j1}", f"C{i}{j2}"
                    restricciones[(var1, var2)] = lambda v1, v2: v1 != v2
        
        # Restricciones de columna
        for j in range(9):
            for i1 in range(9):
                for i2 in range(i1 + 1, 9):
                    var1, var2 = f"C{i1}{j}", f"C{i2}{j}"
                    restricciones[(var1, var2)] = lambda v1, v2: v1 != v2
        
        # Restricciones de subcuadrícula 3x3
        for bloque_i in range(3):
            for bloque_j in range(3):
                celdas_bloque = []
                for i in range(3):
                    for j in range(3):
                        celdas_bloque.append(f"C{bloque_i*3+i}{bloque_j*3+j}")
                
                for idx1 in range(len(celdas_bloque)):
                    for idx2 in range(idx1 + 1, len(celdas_bloque)):
                        var1, var2 = celdas_bloque[idx1], celdas_bloque[idx2]
                        if (var1, var2) not in restricciones:
                            restricciones[(var1, var2)] = lambda v1, v2: v1 != v2
        
        super().__init__(variables, dominios, restricciones)


def visualizar_solucion_reinas(solucion: Dict[str, int], n: int):
    """Visualiza la solución del problema de N-Reinas"""
    if not solucion:
        print("No hay solución")
        return
    
    tablero = [['.' for _ in range(n)] for _ in range(n)]
    for i in range(n):
        fila = solucion[f"Q{i}"]
        tablero[fila][i] = 'Q'
    
    for fila in tablero:
        print(' '.join(fila))


def visualizar_solucion_coloracion(solucion: Dict[str, str], aristas: List[Tuple[str, str]]):
    """Visualiza la solución del problema de coloración"""
    if not solucion:
        print("No hay solución")
        return
    
    print("Asignación de colores:")
    for nodo, color in sorted(solucion.items()):
        print(f"  {nodo}: {color}")
    
    print("\nVerificación de aristas:")
    for nodo1, nodo2 in aristas:
        color1, color2 = solucion[nodo1], solucion[nodo2]
        estado = "✓" if color1 != color2 else "✗"
        print(f"  {estado} {nodo1}({color1}) - {nodo2}({color2})")


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Problemas de Satisfacción de Restricciones (CSP) ===\n")
    
    # Ejemplo 1: 4-Reinas
    print("--- Problema de 4-Reinas ---")
    problema_reinas = ProblemaReinas(4)
    print(f"Variables: {problema_reinas.variables}")
    print(f"Dominio de Q0: {problema_reinas.dominios['Q0']}")
    print(f"Número de restricciones: {len(problema_reinas.restricciones)}")
    print(f"Vecinos de Q0: {problema_reinas.vecinos['Q0']}\n")
    
    # Verificar consistencia
    asignacion_parcial = {"Q0": 1, "Q1": 3}
    print(f"Asignación parcial: {asignacion_parcial}")
    print(f"¿Es consistente Q2=0? {problema_reinas.es_consistente('Q2', 0, asignacion_parcial)}")
    print(f"¿Es consistente Q2=2? {problema_reinas.es_consistente('Q2', 2, asignacion_parcial)}")
    valores_legales = problema_reinas.obtener_valores_legales('Q2', asignacion_parcial)
    print(f"Valores legales para Q2: {valores_legales}\n")
    
    # Ejemplo 2: Coloración de grafo
    print("--- Problema de Coloración de Grafo ---")
    nodos = ["A", "B", "C", "D", "E"]
    aristas = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("C", "E"), ("D", "E")]
    problema_coloracion = ProblemaColoracionGrafo(nodos, aristas, num_colores=3)
    
    print(f"Nodos: {nodos}")
    print(f"Aristas: {aristas}")
    print(f"Colores disponibles: {problema_coloracion.dominios['A']}")
    print(f"Vecinos de C: {problema_coloracion.vecinos['C']}\n")
    
    # Ejemplo 3: Sudoku simple
    print("--- Problema de Sudoku (ejemplo pequeño) ---")
    tablero_inicial = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    problema_sudoku = ProblemaSudoku(tablero_inicial)
    print(f"Variables totales: {len(problema_sudoku.variables)}")
    print(f"Dominio de C00 (5): {problema_sudoku.dominios['C00']}")
    print(f"Dominio de C02 (vacío): {problema_sudoku.dominios['C02']}")
    print(f"Restricciones totales: {len(problema_sudoku.restricciones)}\n")
    
    print("="*50)
    print("\nCaracterísticas de CSP:")
    print("- Formulación declarativa del problema")
    print("- Variables, dominios y restricciones explícitas")
    print("- Permite usar algoritmos de búsqueda especializados")
    print("- Técnicas de inferencia para reducir el espacio de búsqueda")

