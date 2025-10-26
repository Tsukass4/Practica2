"""
Algoritmo 19: Comprobación Hacia Delante (Forward Checking)

Forward Checking mantiene un seguimiento de los valores legales restantes
para variables no asignadas. Cuando se asigna una variable, se eliminan
valores inconsistentes de las variables vecinas.

Ventajas:
- Detecta fallos antes que backtracking simple
- Reduce el factor de ramificación
- Evita asignaciones que llevarán a fallos
"""

from typing import Dict, List, Any, Optional, Set
from copy import deepcopy


class CSP:
    """Clase CSP simplificada para los ejemplos"""
    
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


class SolucionadorForwardChecking:
    """Solucionador CSP con Forward Checking"""
    
    def __init__(self, csp):
        self.csp = csp
        self.asignaciones_probadas = 0
        self.retrocesos = 0
        self.podas = 0
    
    def resolver(self) -> Optional[Dict[str, Any]]:
        """Resuelve el CSP usando forward checking"""
        self.asignaciones_probadas = 0
        self.retrocesos = 0
        self.podas = 0
        
        # Inicializar dominios actuales (copia de los dominios originales)
        dominios_actuales = {var: self.csp.dominios[var].copy() 
                            for var in self.csp.variables}
        
        return self._backtrack({}, dominios_actuales)
    
    def _backtrack(self, asignacion: Dict[str, Any], 
                   dominios: Dict[str, List[Any]]) -> Optional[Dict[str, Any]]:
        """Backtracking con forward checking"""
        
        # Caso base: todas las variables asignadas
        if len(asignacion) == len(self.csp.variables):
            return asignacion
        
        # Seleccionar variable (MRV)
        var = self._seleccionar_variable_mrv(asignacion, dominios)
        
        # Probar cada valor del dominio actual
        for valor in dominios[var]:
            self.asignaciones_probadas += 1
            
            if self.csp.es_consistente(var, valor, asignacion):
                # Asignar valor
                asignacion[var] = valor
                
                # Forward checking: actualizar dominios de vecinos
                dominios_nuevos, exito = self._forward_check(var, valor, asignacion, dominios)
                
                if exito:
                    # Continuar con dominios actualizados
                    resultado = self._backtrack(asignacion, dominios_nuevos)
                    if resultado is not None:
                        return resultado
                
                # Retroceder
                del asignacion[var]
                self.retrocesos += 1
        
        return None
    
    def _forward_check(self, var: str, valor: Any, asignacion: Dict[str, Any],
                      dominios: Dict[str, List[Any]]) -> tuple[Dict[str, List[Any]], bool]:
        """
        Realiza forward checking: elimina valores inconsistentes de vecinos no asignados.
        
        Returns:
            Tupla (nuevos_dominios, exito)
            exito es False si algún dominio queda vacío
        """
        dominios_nuevos = deepcopy(dominios)
        
        # Para cada vecino no asignado
        for vecino in self.csp.vecinos[var]:
            if vecino not in asignacion:
                # Filtrar valores que son inconsistentes con la asignación actual
                valores_validos = []
                for valor_vecino in dominios_nuevos[vecino]:
                    asignacion_temp = asignacion.copy()
                    asignacion_temp[vecino] = valor_vecino
                    
                    if self.csp.es_consistente(vecino, valor_vecino, asignacion_temp):
                        valores_validos.append(valor_vecino)
                    else:
                        self.podas += 1
                
                # Actualizar dominio del vecino
                dominios_nuevos[vecino] = valores_validos
                
                # Si el dominio queda vacío, fallo
                if not valores_validos:
                    return dominios_nuevos, False
        
        return dominios_nuevos, True
    
    def _seleccionar_variable_mrv(self, asignacion: Dict[str, Any],
                                  dominios: Dict[str, List[Any]]) -> str:
        """Selecciona variable con menor dominio (MRV)"""
        variables_no_asignadas = [v for v in self.csp.variables if v not in asignacion]
        return min(variables_no_asignadas, key=lambda v: len(dominios[v]))


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


class ProblemaColoracionGrafo(CSP):
    """Problema de coloración de grafos"""
    
    def __init__(self, nodos, aristas, num_colores=3):
        variables = nodos
        colores = list(range(num_colores))
        dominios = {nodo: colores.copy() for nodo in nodos}
        
        restricciones = {}
        for nodo1, nodo2 in aristas:
            restricciones[(nodo1, nodo2)] = lambda c1, c2: c1 != c2
        
        super().__init__(variables, dominios, restricciones)


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Comprobación Hacia Delante (Forward Checking) ===\n")
    
    # Ejemplo 1: 8-Reinas
    print("--- 8-Reinas con Forward Checking ---")
    problema = ProblemaReinas(8)
    solucionador = SolucionadorForwardChecking(problema)
    solucion = solucionador.resolver()
    
    if solucion:
        print(f"✓ Solución encontrada")
        print(f"Asignaciones probadas: {solucionador.asignaciones_probadas}")
        print(f"Retrocesos: {solucionador.retrocesos}")
        print(f"Podas realizadas: {solucionador.podas}\n")
        
        # Visualizar
        tablero = [['.' for _ in range(8)] for _ in range(8)]
        for i in range(8):
            fila = solucion[f"Q{i}"]
            tablero[fila][i] = 'Q'
        for fila in tablero:
            print(' '.join(fila))
    else:
        print("✗ No se encontró solución")
    
    print("\n" + "="*50 + "\n")
    
    # Ejemplo 2: Coloración de grafo
    print("--- Coloración de Grafo con Forward Checking ---")
    nodos = ["A", "B", "C", "D", "E"]
    aristas = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D"), ("C", "E"), ("D", "E")]
    problema_color = ProblemaColoracionGrafo(nodos, aristas, num_colores=3)
    
    solucionador_color = SolucionadorForwardChecking(problema_color)
    solucion_color = solucionador_color.resolver()
    
    if solucion_color:
        print(f"✓ Solución encontrada")
        print(f"Asignaciones probadas: {solucionador_color.asignaciones_probadas}")
        print(f"Retrocesos: {solucionador_color.retrocesos}")
        print(f"Podas realizadas: {solucionador_color.podas}\n")
        
        print("Asignación de colores:")
        for nodo in sorted(solucion_color.keys()):
            print(f"  {nodo}: Color {solucion_color[nodo]}")
    else:
        print("✗ No se encontró solución")
    
    print("\n" + "="*50 + "\n")
    
    # Comparación con backtracking simple
    print("--- Comparación: Backtracking vs Forward Checking ---")
    
    # Backtracking simple (sin FC)
    class SolucionadorSimple:
        def __init__(self, csp):
            self.csp = csp
            self.asignaciones_probadas = 0
        
        def resolver(self):
            self.asignaciones_probadas = 0
            return self._backtrack({})
        
        def _backtrack(self, asignacion):
            if len(asignacion) == len(self.csp.variables):
                return asignacion
            
            var = [v for v in self.csp.variables if v not in asignacion][0]
            
            for valor in self.csp.dominios[var]:
                self.asignaciones_probadas += 1
                if self.csp.es_consistente(var, valor, asignacion):
                    asignacion[var] = valor
                    resultado = self._backtrack(asignacion)
                    if resultado:
                        return resultado
                    del asignacion[var]
            return None
    
    problema_test = ProblemaReinas(6)
    
    sol_simple = SolucionadorSimple(problema_test)
    sol_simple.resolver()
    
    sol_fc = SolucionadorForwardChecking(problema_test)
    sol_fc.resolver()
    
    print(f"6-Reinas:")
    print(f"  Backtracking simple: {sol_simple.asignaciones_probadas} asignaciones")
    print(f"  Forward Checking: {sol_fc.asignaciones_probadas} asignaciones")
    print(f"  Podas con FC: {sol_fc.podas}")
    
    mejora = ((sol_simple.asignaciones_probadas - sol_fc.asignaciones_probadas) / 
              sol_simple.asignaciones_probadas * 100)
    print(f"  Mejora: {mejora:.1f}% menos asignaciones")
    
    print("\n" + "="*50)
    print("\nCaracterísticas de Forward Checking:")
    print("- Detecta fallos antes que backtracking simple")
    print("- Mantiene dominios actualizados para variables no asignadas")
    print("- Poda valores inconsistentes inmediatamente")
    print("- Reduce significativamente el espacio de búsqueda")
    print("- Más eficiente que backtracking simple en la mayoría de casos")

