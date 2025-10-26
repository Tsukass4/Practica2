"""
Algoritmo 20: Propagación de Restricciones (Constraint Propagation)

La propagación de restricciones usa restricciones para reducir dominios.
El algoritmo más común es AC-3 (Arc Consistency 3).

AC-3 hace que el CSP sea arco-consistente:
- Un arco (Xi, Xj) es consistente si para cada valor en Di existe
  un valor en Dj que satisface la restricción.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from collections import deque
from copy import deepcopy


class CSP:
    """Clase CSP para propagación de restricciones"""
    
    def __init__(self, variables, dominios, restricciones):
        self.variables = variables
        self.dominios = dominios
        self.restricciones = restricciones
        self.vecinos = self._calcular_vecinos()
        self.arcos = self._calcular_arcos()
    
    def _calcular_vecinos(self):
        vecinos = {var: set() for var in self.variables}
        for variables_restriccion in self.restricciones.keys():
            for var1 in variables_restriccion:
                for var2 in variables_restriccion:
                    if var1 != var2:
                        vecinos[var1].add(var2)
        return vecinos
    
    def _calcular_arcos(self):
        """Calcula todos los arcos del CSP"""
        arcos = []
        for (var1, var2) in self.restricciones.keys():
            arcos.append((var1, var2))
            arcos.append((var2, var1))
        return arcos
    
    def es_consistente_binaria(self, var1, valor1, var2, valor2):
        """Verifica si dos valores son consistentes"""
        if (var1, var2) in self.restricciones:
            return self.restricciones[(var1, var2)](valor1, valor2)
        elif (var2, var1) in self.restricciones:
            return self.restricciones[(var2, var1)](valor2, valor1)
        return True


def ac3(csp: CSP, cola_arcos: Optional[List[Tuple[str, str]]] = None) -> Tuple[bool, Dict[str, List[Any]]]:
    """
    Algoritmo AC-3 para consistencia de arcos.
    
    Args:
        csp: El CSP
        cola_arcos: Cola inicial de arcos (None = todos los arcos)
    
    Returns:
        Tupla (exito, dominios_reducidos)
        exito es False si algún dominio queda vacío
    """
    # Copiar dominios
    dominios = {var: csp.dominios[var].copy() for var in csp.variables}
    
    # Inicializar cola de arcos
    if cola_arcos is None:
        cola = deque(csp.arcos)
    else:
        cola = deque(cola_arcos)
    
    revisiones = 0
    podas = 0
    
    while cola:
        (xi, xj) = cola.popleft()
        revisiones += 1
        
        # Revisar arco
        revisado, num_podas = revisar_arco(csp, xi, xj, dominios)
        podas += num_podas
        
        if revisado:
            # Si el dominio de Xi quedó vacío, fallo
            if not dominios[xi]:
                return False, dominios
            
            # Agregar arcos (Xk, Xi) para todos los vecinos Xk de Xi (excepto Xj)
            for xk in csp.vecinos[xi]:
                if xk != xj:
                    cola.append((xk, xi))
    
    print(f"  AC-3: {revisiones} revisiones, {podas} valores podados")
    return True, dominios


def revisar_arco(csp: CSP, xi: str, xj: str, dominios: Dict[str, List[Any]]) -> Tuple[bool, int]:
    """
    Revisa el arco (Xi, Xj) y elimina valores inconsistentes de Di.
    
    Returns:
        Tupla (revisado, num_podas)
        revisado es True si se eliminó algún valor
    """
    revisado = False
    num_podas = 0
    valores_a_eliminar = []
    
    # Para cada valor en el dominio de Xi
    for valor_i in dominios[xi]:
        # Verificar si existe algún valor en Dj que sea consistente
        existe_consistente = False
        
        for valor_j in dominios[xj]:
            if csp.es_consistente_binaria(xi, valor_i, xj, valor_j):
                existe_consistente = True
                break
        
        # Si no existe valor consistente, eliminar valor_i
        if not existe_consistente:
            valores_a_eliminar.append(valor_i)
            revisado = True
            num_podas += 1
    
    # Eliminar valores
    for valor in valores_a_eliminar:
        dominios[xi].remove(valor)
    
    return revisado, num_podas


class SolucionadorMAC:
    """Solucionador con Maintaining Arc Consistency (MAC)"""
    
    def __init__(self, csp):
        self.csp = csp
        self.asignaciones_probadas = 0
        self.retrocesos = 0
        self.podas_ac3 = 0
    
    def resolver(self) -> Optional[Dict[str, Any]]:
        """Resuelve el CSP usando MAC"""
        self.asignaciones_probadas = 0
        self.retrocesos = 0
        
        # Preprocesamiento: AC-3 inicial
        print("Preprocesamiento con AC-3...")
        exito, dominios_iniciales = ac3(self.csp)
        
        if not exito:
            print("  ✗ CSP inconsistente detectado en preprocesamiento")
            return None
        
        print(f"  ✓ Dominios reducidos")
        for var in self.csp.variables[:3]:  # Mostrar primeras 3 variables
            print(f"    {var}: {len(self.csp.dominios[var])} -> {len(dominios_iniciales[var])} valores")
        
        return self._backtrack({}, dominios_iniciales)
    
    def _backtrack(self, asignacion: Dict[str, Any], 
                   dominios: Dict[str, List[Any]]) -> Optional[Dict[str, Any]]:
        """Backtracking con MAC"""
        
        if len(asignacion) == len(self.csp.variables):
            return asignacion
        
        # Seleccionar variable (MRV)
        var = self._seleccionar_variable_mrv(asignacion, dominios)
        
        for valor in dominios[var]:
            self.asignaciones_probadas += 1
            
            # Asignar valor
            asignacion[var] = valor
            
            # Reducir dominios de vecinos
            dominios_nuevos = deepcopy(dominios)
            dominios_nuevos[var] = [valor]
            
            # Aplicar AC-3 a los arcos afectados
            arcos_afectados = [(vecino, var) for vecino in self.csp.vecinos[var]
                              if vecino not in asignacion]
            
            exito, dominios_nuevos = ac3(self.csp, arcos_afectados)
            
            if exito:
                resultado = self._backtrack(asignacion, dominios_nuevos)
                if resultado is not None:
                    return resultado
            
            # Retroceder
            del asignacion[var]
            self.retrocesos += 1
        
        return None
    
    def _seleccionar_variable_mrv(self, asignacion: Dict[str, Any],
                                  dominios: Dict[str, List[Any]]) -> str:
        """Selecciona variable con menor dominio"""
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


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Propagación de Restricciones (AC-3 y MAC) ===\n")
    
    # Ejemplo 1: AC-3 en 4-Reinas
    print("--- AC-3 en 4-Reinas ---")
    problema = ProblemaReinas(4)
    
    print(f"Dominios iniciales:")
    for var in problema.variables:
        print(f"  {var}: {problema.dominios[var]}")
    
    print(f"\nAplicando AC-3...")
    exito, dominios_reducidos = ac3(problema)
    
    if exito:
        print(f"\n✓ CSP es arco-consistente")
        print(f"Dominios después de AC-3:")
        for var in problema.variables:
            print(f"  {var}: {dominios_reducidos[var]}")
    else:
        print(f"\n✗ CSP inconsistente")
    
    print("\n" + "="*50 + "\n")
    
    # Ejemplo 2: MAC en 8-Reinas
    print("--- MAC (Maintaining Arc Consistency) en 8-Reinas ---\n")
    problema8 = ProblemaReinas(8)
    solucionador = SolucionadorMAC(problema8)
    solucion = solucionador.resolver()
    
    if solucion:
        print(f"\n✓ Solución encontrada")
        print(f"Asignaciones probadas: {solucionador.asignaciones_probadas}")
        print(f"Retrocesos: {solucionador.retrocesos}\n")
        
        # Visualizar
        tablero = [['.' for _ in range(8)] for _ in range(8)]
        for i in range(8):
            fila = solucion[f"Q{i}"]
            tablero[fila][i] = 'Q'
        for fila in tablero:
            print(' '.join(fila))
    else:
        print(f"\n✗ No se encontró solución")
    
    print("\n" + "="*50)
    print("\nCaracterísticas de la Propagación de Restricciones:")
    print("- AC-3: Hace el CSP arco-consistente")
    print("- Reduce dominios antes de la búsqueda")
    print("- Detecta inconsistencias tempranamente")
    print("- MAC: Mantiene arco-consistencia durante backtracking")
    print("- Más costoso por iteración pero menos iteraciones totales")
    print("\nComplejidad de AC-3:")
    print("- Tiempo: O(ed³) donde e=arcos, d=tamaño de dominio")
    print("- Espacio: O(e) para la cola")

