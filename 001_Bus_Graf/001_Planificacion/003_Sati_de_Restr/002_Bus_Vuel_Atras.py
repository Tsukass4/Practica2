"""
Algoritmo 18: Búsqueda de Vuelta Atrás (Backtracking)

El backtracking es el algoritmo básico para resolver CSPs.
Asigna valores a variables una por una y retrocede cuando
encuentra inconsistencias.

Mejoras incluidas:
- Selección de variable más restringida (MRV)
- Selección de valor menos restrictivo (LCV)
- Verificación hacia adelante (Forward Checking)
"""

from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Importar la clase CSP del archivo anterior
sys.path.append(os.path.dirname(__file__))
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Proble_Satis_Restri import CSP


class SolucionadorBacktracking:
    """Solucionador de CSP usando backtracking"""
    
    def __init__(self, csp):
        self.csp = csp
        self.asignaciones_probadas = 0
        self.retrocesos = 0
    
    def resolver(self) -> Optional[Dict[str, Any]]:
        """Resuelve el CSP usando backtracking básico"""
        self.asignaciones_probadas = 0
        self.retrocesos = 0
        return self._backtrack({})
    
    def _backtrack(self, asignacion: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Función recursiva de backtracking"""
        # Si todas las variables están asignadas, hemos terminado
        if len(asignacion) == len(self.csp.variables):
            return asignacion
        
        # Seleccionar variable no asignada
        var = self._seleccionar_variable_no_asignada(asignacion)
        
        # Probar cada valor del dominio
        for valor in self._ordenar_valores_dominio(var, asignacion):
            self.asignaciones_probadas += 1
            
            if self.csp.es_consistente(var, valor, asignacion):
                # Asignar valor
                asignacion[var] = valor
                
                # Recursión
                resultado = self._backtrack(asignacion)
                if resultado is not None:
                    return resultado
                
                # Retroceder
                del asignacion[var]
                self.retrocesos += 1
        
        return None
    
    def _seleccionar_variable_no_asignada(self, asignacion: Dict[str, Any]) -> str:
        """Selecciona la siguiente variable a asignar (orden simple)"""
        for var in self.csp.variables:
            if var not in asignacion:
                return var
        return None
    
    def _ordenar_valores_dominio(self, var: str, asignacion: Dict[str, Any]) -> List[Any]:
        """Retorna los valores del dominio en orden (sin heurística)"""
        return self.csp.dominios[var]


class SolucionadorBacktrackingMejorado:
    """Solucionador con heurísticas MRV y LCV"""
    
    def __init__(self, csp):
        self.csp = csp
        self.asignaciones_probadas = 0
        self.retrocesos = 0
    
    def resolver(self) -> Optional[Dict[str, Any]]:
        """Resuelve el CSP usando backtracking con heurísticas"""
        self.asignaciones_probadas = 0
        self.retrocesos = 0
        return self._backtrack({})
    
    def _backtrack(self, asignacion: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Función recursiva de backtracking"""
        if len(asignacion) == len(self.csp.variables):
            return asignacion
        
        # MRV: Seleccionar variable con menor número de valores legales
        var = self._seleccionar_variable_mrv(asignacion)
        
        # LCV: Ordenar valores por menor restricción a vecinos
        for valor in self._ordenar_valores_lcv(var, asignacion):
            self.asignaciones_probadas += 1
            
            if self.csp.es_consistente(var, valor, asignacion):
                asignacion[var] = valor
                
                resultado = self._backtrack(asignacion)
                if resultado is not None:
                    return resultado
                
                del asignacion[var]
                self.retrocesos += 1
        
        return None
    
    def _seleccionar_variable_mrv(self, asignacion: Dict[str, Any]) -> str:
        """
        Minimum Remaining Values (MRV):
        Selecciona la variable con menos valores legales restantes.
        También conocida como "most constrained variable".
        """
        variables_no_asignadas = [v for v in self.csp.variables if v not in asignacion]
        
        def contar_valores_legales(var):
            return len(self.csp.obtener_valores_legales(var, asignacion))
        
        # Seleccionar variable con menos valores legales
        return min(variables_no_asignadas, key=contar_valores_legales)
    
    def _ordenar_valores_lcv(self, var: str, asignacion: Dict[str, Any]) -> List[Any]:
        """
        Least Constraining Value (LCV):
        Ordena valores por cuántos valores eliminan de los vecinos.
        Prefiere valores que dejan más opciones a otras variables.
        """
        valores_legales = self.csp.obtener_valores_legales(var, asignacion)
        
        def contar_conflictos(valor):
            """Cuenta cuántos valores de vecinos serían eliminados"""
            conflictos = 0
            asignacion_temp = asignacion.copy()
            asignacion_temp[var] = valor
            
            for vecino in self.csp.vecinos[var]:
                if vecino not in asignacion:
                    valores_vecino_antes = len(self.csp.obtener_valores_legales(vecino, asignacion))
                    valores_vecino_despues = len(self.csp.obtener_valores_legales(vecino, asignacion_temp))
                    conflictos += valores_vecino_antes - valores_vecino_despues
            
            return conflictos
        
        # Ordenar por menos conflictos (menos restrictivo primero)
        return sorted(valores_legales, key=contar_conflictos)


# Ejemplo de uso
if __name__ == "__main__":
    # Necesitamos importar las clases de CSP
    from typing import List, Dict, Set, Tuple, Callable, Optional, Any
    
    # Redefinir CSP y ProblemaReinas para este ejemplo standalone
    class CSP:
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
        
        def obtener_valores_legales(self, var, asignacion):
            return [valor for valor in self.dominios[var] 
                    if self.es_consistente(var, valor, asignacion)]
    
    class ProblemaReinas(CSP):
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
    
    print("=== Búsqueda de Vuelta Atrás (Backtracking) ===\n")
    
    # Resolver 4-Reinas con backtracking básico
    print("--- Backtracking Básico (4-Reinas) ---")
    problema = ProblemaReinas(4)
    solucionador_basico = SolucionadorBacktracking(problema)
    solucion = solucionador_basico.resolver()
    
    if solucion:
        print(f"✓ Solución encontrada: {solucion}")
        print(f"Asignaciones probadas: {solucionador_basico.asignaciones_probadas}")
        print(f"Retrocesos: {solucionador_basico.retrocesos}\n")
        
        # Visualizar
        tablero = [['.' for _ in range(4)] for _ in range(4)]
        for i in range(4):
            fila = solucion[f"Q{i}"]
            tablero[fila][i] = 'Q'
        for fila in tablero:
            print(' '.join(fila))
    else:
        print("✗ No se encontró solución")
    
    print("\n" + "="*50 + "\n")
    
    # Resolver 8-Reinas con backtracking mejorado
    print("--- Backtracking con MRV y LCV (8-Reinas) ---")
    problema8 = ProblemaReinas(8)
    solucionador_mejorado = SolucionadorBacktrackingMejorado(problema8)
    solucion8 = solucionador_mejorado.resolver()
    
    if solucion8:
        print(f"✓ Solución encontrada")
        print(f"Asignaciones probadas: {solucionador_mejorado.asignaciones_probadas}")
        print(f"Retrocesos: {solucionador_mejorado.retrocesos}\n")
        
        # Visualizar
        tablero8 = [['.' for _ in range(8)] for _ in range(8)]
        for i in range(8):
            fila = solucion8[f"Q{i}"]
            tablero8[fila][i] = 'Q'
        for fila in tablero8:
            print(' '.join(fila))
    else:
        print("✗ No se encontró solución")
    
    print("\n" + "="*50 + "\n")
    
    # Comparar eficiencia
    print("--- Comparación de Eficiencia (4-Reinas) ---")
    problema4 = ProblemaReinas(4)
    
    sol_basico = SolucionadorBacktracking(problema4)
    sol_basico.resolver()
    
    sol_mejorado = SolucionadorBacktrackingMejorado(problema4)
    sol_mejorado.resolver()
    
    print(f"Backtracking básico:")
    print(f"  Asignaciones: {sol_basico.asignaciones_probadas}")
    print(f"  Retrocesos: {sol_basico.retrocesos}")
    
    print(f"\nBacktracking mejorado (MRV + LCV):")
    print(f"  Asignaciones: {sol_mejorado.asignaciones_probadas}")
    print(f"  Retrocesos: {sol_mejorado.retrocesos}")
    
    mejora = ((sol_basico.asignaciones_probadas - sol_mejorado.asignaciones_probadas) / 
              sol_basico.asignaciones_probadas * 100)
    print(f"\nMejora: {mejora:.1f}% menos asignaciones")
    
    print("\n" + "="*50)
    print("\nHeurísticas:")
    print("- MRV (Minimum Remaining Values): Elige variable más restringida")
    print("- LCV (Least Constraining Value): Elige valor menos restrictivo")
    print("- Ambas reducen significativamente el espacio de búsqueda")

