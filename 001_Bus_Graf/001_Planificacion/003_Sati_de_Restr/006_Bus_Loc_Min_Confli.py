"""
Algoritmo 22: Búsqueda Local - Mínimos Conflictos (Min-Conflicts)

Min-Conflicts es un algoritmo de búsqueda local para CSPs.
En cada paso, selecciona una variable en conflicto y le asigna
el valor que minimiza el número de conflictos.

Características:
- Muy eficiente para problemas grandes
- No garantiza encontrar solución
- Útil para problemas casi-resueltos
- Usado en scheduling y planificación
"""

import random
from typing import Dict, List, Any, Optional, Set


class CSP:
    """Clase CSP para Min-Conflicts"""
    
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
    
    def contar_conflictos(self, var, valor, asignacion):
        """Cuenta el número de conflictos si var=valor"""
        conflictos = 0
        asignacion_temp = asignacion.copy()
        asignacion_temp[var] = valor
        
        for variables_restriccion, funcion_restriccion in self.restricciones.items():
            if var in variables_restriccion:
                if all(v in asignacion_temp for v in variables_restriccion):
                    valores = tuple(asignacion_temp[v] for v in variables_restriccion)
                    if not funcion_restriccion(*valores):
                        conflictos += 1
        
        return conflictos
    
    def obtener_variables_en_conflicto(self, asignacion):
        """Retorna las variables que tienen conflictos"""
        variables_conflicto = set()
        
        for variables_restriccion, funcion_restriccion in self.restricciones.items():
            if all(v in asignacion for v in variables_restriccion):
                valores = tuple(asignacion[v] for v in variables_restriccion)
                if not funcion_restriccion(*valores):
                    variables_conflicto.update(variables_restriccion)
        
        return list(variables_conflicto)


def min_conflicts(csp: CSP, max_pasos: int = 1000) -> Optional[Dict[str, Any]]:
    """
    Algoritmo Min-Conflicts para resolver CSPs.
    
    Args:
        csp: El CSP a resolver
        max_pasos: Número máximo de pasos
    
    Returns:
        Asignación completa si tiene éxito, None si falla
    """
    # Generar asignación inicial aleatoria
    asignacion = {var: random.choice(csp.dominios[var]) for var in csp.variables}
    
    for paso in range(max_pasos):
        # Obtener variables en conflicto
        variables_conflicto = csp.obtener_variables_en_conflicto(asignacion)
        
        # Si no hay conflictos, hemos terminado
        if not variables_conflicto:
            return asignacion
        
        # Seleccionar aleatoriamente una variable en conflicto
        var = random.choice(variables_conflicto)
        
        # Encontrar el valor que minimiza conflictos
        mejor_valor = None
        min_conflictos = float('inf')
        valores_minimos = []
        
        for valor in csp.dominios[var]:
            num_conflictos = csp.contar_conflictos(var, valor, asignacion)
            
            if num_conflictos < min_conflictos:
                min_conflictos = num_conflictos
                valores_minimos = [valor]
            elif num_conflictos == min_conflictos:
                valores_minimos.append(valor)
        
        # Elegir aleatoriamente entre los valores que minimizan conflictos
        asignacion[var] = random.choice(valores_minimos)
    
    # No se encontró solución en max_pasos
    return None


def min_conflicts_con_estadisticas(csp: CSP, max_pasos: int = 1000) -> tuple:
    """Versión de Min-Conflicts que retorna estadísticas"""
    asignacion = {var: random.choice(csp.dominios[var]) for var in csp.variables}
    
    historial_conflictos = []
    cambios = 0
    
    for paso in range(max_pasos):
        variables_conflicto = csp.obtener_variables_en_conflicto(asignacion)
        num_conflictos = len(variables_conflicto)
        historial_conflictos.append(num_conflictos)
        
        if not variables_conflicto:
            return asignacion, cambios, historial_conflictos
        
        var = random.choice(variables_conflicto)
        
        mejor_valor = None
        min_conflictos_local = float('inf')
        valores_minimos = []
        
        for valor in csp.dominios[var]:
            num_conflictos_local = csp.contar_conflictos(var, valor, asignacion)
            
            if num_conflictos_local < min_conflictos_local:
                min_conflictos_local = num_conflictos_local
                valores_minimos = [valor]
            elif num_conflictos_local == min_conflictos_local:
                valores_minimos.append(valor)
        
        nuevo_valor = random.choice(valores_minimos)
        if asignacion[var] != nuevo_valor:
            cambios += 1
        asignacion[var] = nuevo_valor
    
    return None, cambios, historial_conflictos


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
    print("=== Búsqueda Local: Mínimos Conflictos (Min-Conflicts) ===\n")
    
    # Ejemplo 1: 8-Reinas
    print("--- 8-Reinas con Min-Conflicts ---")
    problema = ProblemaReinas(8)
    solucion = min_conflicts(problema, max_pasos=1000)
    
    if solucion:
        print(f"✓ Solución encontrada\n")
        
        # Visualizar
        tablero = [['.' for _ in range(8)] for _ in range(8)]
        for i in range(8):
            fila = solucion[f"Q{i}"]
            tablero[fila][i] = 'Q'
        for fila in tablero:
            print(' '.join(fila))
    else:
        print("✗ No se encontró solución en el límite de pasos")
    
    print("\n" + "="*50 + "\n")
    
    # Ejemplo 2: Estadísticas detalladas
    print("--- Min-Conflicts con Estadísticas (100-Reinas) ---")
    problema_grande = ProblemaReinas(100)
    solucion, cambios, historial = min_conflicts_con_estadisticas(
        problema_grande, max_pasos=10000
    )
    
    if solucion:
        print(f"✓ Solución encontrada para 100-Reinas")
        print(f"Cambios realizados: {cambios}")
        print(f"Pasos hasta solución: {len(historial)}")
        print(f"\nProgreso de conflictos:")
        print(f"  Inicial: {historial[0]} conflictos")
        if len(historial) > 10:
            print(f"  Paso 10: {historial[10]} conflictos")
        if len(historial) > 100:
            print(f"  Paso 100: {historial[100]} conflictos")
        print(f"  Final: {historial[-1]} conflictos")
    else:
        print(f"✗ No se encontró solución")
        print(f"Cambios realizados: {cambios}")
        print(f"Conflictos finales: {historial[-1]}")
    
    print("\n" + "="*50 + "\n")
    
    # Ejemplo 3: Múltiples ejecuciones
    print("--- Tasa de Éxito (10 ejecuciones de 8-Reinas) ---")
    exitos = 0
    total_pasos = 0
    
    for i in range(10):
        solucion_i, cambios_i, historial_i = min_conflicts_con_estadisticas(
            ProblemaReinas(8), max_pasos=1000
        )
        if solucion_i:
            exitos += 1
            total_pasos += len(historial_i)
    
    print(f"Soluciones encontradas: {exitos}/10")
    if exitos > 0:
        print(f"Pasos promedio: {total_pasos/exitos:.1f}")
    
    print("\n" + "="*50)
    print("\nCaracterísticas de Min-Conflicts:")
    print("- Búsqueda local: Comienza con asignación completa")
    print("- Greedy: Siempre elige el valor con menos conflictos")
    print("- Muy eficiente para problemas grandes")
    print("- No completo: Puede quedar atrapado en mínimos locales")
    print("\nVentajas:")
    print("- Escala muy bien (O(n) para N-Reinas)")
    print("- Útil para problemas casi-resueltos")
    print("- Simple de implementar")
    print("\nDesventajas:")
    print("- No garantiza encontrar solución")
    print("- Puede requerir múltiples reinicios")

