"""
Algoritmo 21: Salto Atrás Dirigido por Conflictos (Conflict-Directed Backjumping)

El backjumping inteligente salta directamente a la variable que causó el conflicto,
en lugar de retroceder cronológicamente. Mantiene un "conjunto de conflictos"
para cada variable.

Ventajas:
- Evita trabajo redundante
- Salta sobre variables irrelevantes
- Más eficiente que backtracking cronológico
"""

from typing import Dict, List, Any, Optional, Set
from copy import deepcopy


class CSP:
    """Clase CSP simplificada"""
    
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
    
    def obtener_variables_conflicto(self, var, valor, asignacion):
        """Retorna las variables asignadas que causan conflicto con var=valor"""
        conflictos = set()
        asignacion_temp = asignacion.copy()
        asignacion_temp[var] = valor
        
        for variables_restriccion, funcion_restriccion in self.restricciones.items():
            if var in variables_restriccion:
                if all(v in asignacion_temp for v in variables_restriccion):
                    valores = tuple(asignacion_temp[v] for v in variables_restriccion)
                    if not funcion_restriccion(*valores):
                        # Agregar las otras variables de la restricción
                        for v in variables_restriccion:
                            if v != var and v in asignacion:
                                conflictos.add(v)
        
        return conflictos


class SolucionadorBackjumping:
    """Solucionador con Conflict-Directed Backjumping"""
    
    def __init__(self, csp):
        self.csp = csp
        self.asignaciones_probadas = 0
        self.retrocesos = 0
        self.saltos = 0
    
    def resolver(self) -> Optional[Dict[str, Any]]:
        """Resuelve el CSP usando backjumping"""
        self.asignaciones_probadas = 0
        self.retrocesos = 0
        self.saltos = 0
        
        # Conjunto de conflictos para cada variable
        conjuntos_conflicto = {var: set() for var in self.csp.variables}
        
        resultado = self._backjump({}, conjuntos_conflicto, 0)
        return resultado if resultado != "FALLO" else None
    
    def _backjump(self, asignacion: Dict[str, Any], 
                  conjuntos_conflicto: Dict[str, Set[str]],
                  nivel: int) -> Any:
        """
        Backtracking con backjumping.
        
        Returns:
            asignacion si tiene éxito
            conjunto_conflicto si falla (para propagar conflictos)
            "FALLO" si no hay solución
        """
        # Caso base: todas las variables asignadas
        if len(asignacion) == len(self.csp.variables):
            return asignacion
        
        # Seleccionar variable
        var = self._seleccionar_variable(asignacion)
        conjuntos_conflicto[var] = set()
        
        # Probar cada valor del dominio
        for valor in self.csp.dominios[var]:
            self.asignaciones_probadas += 1
            
            if self.csp.es_consistente(var, valor, asignacion):
                # Asignar valor
                asignacion[var] = valor
                
                # Recursión
                resultado = self._backjump(asignacion, conjuntos_conflicto, nivel + 1)
                
                if isinstance(resultado, dict):
                    # Éxito
                    return resultado
                elif isinstance(resultado, set):
                    # Fallo: propagar conjunto de conflictos
                    conjunto_conf = resultado
                    
                    if var not in conjunto_conf:
                        # Saltar sobre var
                        del asignacion[var]
                        self.saltos += 1
                        return conjunto_conf
                    else:
                        # var está en el conjunto de conflictos
                        conjunto_conf.remove(var)
                        conjuntos_conflicto[var] = conjuntos_conflicto[var].union(conjunto_conf)
                
                # Retroceder
                del asignacion[var]
                self.retrocesos += 1
            else:
                # Valor inconsistente: agregar variables conflictivas
                conflictos = self.csp.obtener_variables_conflicto(var, valor, asignacion)
                conjuntos_conflicto[var] = conjuntos_conflicto[var].union(conflictos)
        
        # No hay valores válidos para var
        return conjuntos_conflicto[var] if conjuntos_conflicto[var] else "FALLO"
    
    def _seleccionar_variable(self, asignacion: Dict[str, Any]) -> str:
        """Selecciona la siguiente variable a asignar"""
        for var in self.csp.variables:
            if var not in asignacion:
                return var
        return None


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
    print("=== Salto Atrás Dirigido por Conflictos (Backjumping) ===\n")
    
    # Resolver 8-Reinas con backjumping
    print("--- 8-Reinas con Backjumping ---")
    problema = ProblemaReinas(8)
    solucionador = SolucionadorBackjumping(problema)
    solucion = solucionador.resolver()
    
    if solucion:
        print(f"✓ Solución encontrada")
        print(f"Asignaciones probadas: {solucionador.asignaciones_probadas}")
        print(f"Retrocesos: {solucionador.retrocesos}")
        print(f"Saltos realizados: {solucionador.saltos}\n")
        
        # Visualizar
        tablero = [['.' for _ in range(8)] for _ in range(8)]
        for i in range(8):
            fila = solucion[f"Q{i}"]
            tablero[fila][i] = 'Q'
        for fila in tablero:
            print(' '.join(fila))
    else:
        print("✗ No se encontró solución")
    
    print("\n" + "="*50)
    print("\nCaracterísticas del Backjumping:")
    print("- Mantiene conjuntos de conflictos para cada variable")
    print("- Salta directamente a la variable que causó el conflicto")
    print("- Evita retrocesos cronológicos innecesarios")
    print("- Más eficiente cuando hay variables independientes")
    print("\nDiferencias con Backtracking:")
    print("- Backtracking: Retrocede a la variable anterior")
    print("- Backjumping: Salta a la variable conflictiva")

