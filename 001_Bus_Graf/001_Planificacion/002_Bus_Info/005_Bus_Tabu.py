"""
Algoritmo 12: Búsqueda Tabú (Tabu Search)

La búsqueda tabú es una metaheurística que mejora la búsqueda local mediante
el uso de memoria. Mantiene una lista de movimientos prohibidos (tabú) para
evitar ciclos y escapar de óptimos locales.

Características:
- Usa memoria a corto plazo (lista tabú)
- Puede aceptar movimientos que empeoran la solución
- Evita ciclos mediante restricciones tabú
- Criterio de aspiración para anular restricciones
"""

import random
from typing import List, Tuple, Set, Optional
from collections import deque


class ProblemaReinas:
    """Problema de las N-Reinas"""
    
    def __init__(self, n: int = 8):
        self.n = n
    
    def evaluar(self, estado: List[int]) -> int:
        """
        Cuenta el número de pares de reinas que se atacan.
        Menor valor = mejor estado (0 = solución).
        """
        ataques = 0
        for i in range(len(estado)):
            for j in range(i + 1, len(estado)):
                # Misma fila
                if estado[i] == estado[j]:
                    ataques += 1
                # Misma diagonal
                elif abs(estado[i] - estado[j]) == abs(i - j):
                    ataques += 1
        return ataques
    
    def obtener_vecinos(self, estado: List[int]) -> List[Tuple[List[int], Tuple[int, int]]]:
        """
        Genera vecinos y sus movimientos.
        Retorna lista de tuplas (vecino, movimiento)
        donde movimiento = (columna, nueva_fila)
        """
        vecinos = []
        for col in range(self.n):
            for fila in range(self.n):
                if fila != estado[col]:
                    vecino = estado.copy()
                    vecino[col] = fila
                    movimiento = (col, fila)
                    vecinos.append((vecino, movimiento))
        return vecinos
    
    def estado_aleatorio(self) -> List[int]:
        """Genera un estado aleatorio"""
        return [random.randint(0, self.n - 1) for _ in range(self.n)]
    
    def es_solucion(self, estado: List[int]) -> bool:
        """Verifica si es una solución válida"""
        return self.evaluar(estado) == 0


class ListaTabu:
    """Maneja la lista de movimientos tabú"""
    
    def __init__(self, tamano_maximo: int):
        self.tamano_maximo = tamano_maximo
        self.lista = deque(maxlen=tamano_maximo)
    
    def agregar(self, movimiento: Tuple[int, int]):
        """Agrega un movimiento a la lista tabú"""
        self.lista.append(movimiento)
    
    def es_tabu(self, movimiento: Tuple[int, int]) -> bool:
        """Verifica si un movimiento es tabú"""
        return movimiento in self.lista
    
    def limpiar(self):
        """Limpia la lista tabú"""
        self.lista.clear()


def busqueda_tabu(problema: ProblemaReinas,
                  estado_inicial: List[int],
                  tamano_tabu: int = 10,
                  max_iteraciones: int = 1000,
                  max_sin_mejora: int = 50) -> Tuple[List[int], int, int, List[int]]:
    """
    Implementación de la búsqueda tabú.
    
    Args:
        problema: El problema a resolver
        estado_inicial: Estado desde donde comenzar
        tamano_tabu: Tamaño de la lista tabú
        max_iteraciones: Máximo número de iteraciones
        max_sin_mejora: Iteraciones sin mejora antes de terminar
    
    Returns:
        Tupla con (mejor_estado, mejor_valor, iteraciones, historial_valores)
    """
    # Inicializar
    estado_actual = estado_inicial
    valor_actual = problema.evaluar(estado_actual)
    
    mejor_estado = estado_actual.copy()
    mejor_valor = valor_actual
    
    lista_tabu = ListaTabu(tamano_tabu)
    historial_valores = [valor_actual]
    
    iteraciones = 0
    iteraciones_sin_mejora = 0
    
    while iteraciones < max_iteraciones and iteraciones_sin_mejora < max_sin_mejora:
        iteraciones += 1
        
        # Si encontramos la solución óptima, terminar
        if mejor_valor == 0:
            break
        
        # Obtener todos los vecinos
        vecinos = problema.obtener_vecinos(estado_actual)
        
        # Encontrar el mejor vecino no tabú (o que cumpla criterio de aspiración)
        mejor_vecino = None
        mejor_valor_vecino = float('inf')
        mejor_movimiento = None
        
        for vecino, movimiento in vecinos:
            valor_vecino = problema.evaluar(vecino)
            
            # Criterio de aspiración: aceptar movimiento tabú si mejora el mejor global
            es_tabu = lista_tabu.es_tabu(movimiento)
            criterio_aspiracion = valor_vecino < mejor_valor
            
            if (not es_tabu or criterio_aspiracion) and valor_vecino < mejor_valor_vecino:
                mejor_vecino = vecino
                mejor_valor_vecino = valor_vecino
                mejor_movimiento = movimiento
        
        # Si no hay vecinos válidos, terminar
        if mejor_vecino is None:
            break
        
        # Moverse al mejor vecino
        estado_actual = mejor_vecino
        valor_actual = mejor_valor_vecino
        
        # Agregar movimiento a la lista tabú
        lista_tabu.agregar(mejor_movimiento)
        
        # Actualizar mejor solución global
        if valor_actual < mejor_valor:
            mejor_estado = estado_actual.copy()
            mejor_valor = valor_actual
            iteraciones_sin_mejora = 0
        else:
            iteraciones_sin_mejora += 1
        
        historial_valores.append(valor_actual)
    
    return mejor_estado, mejor_valor, iteraciones, historial_valores


def busqueda_tabu_adaptativa(problema: ProblemaReinas,
                             estado_inicial: List[int],
                             tamano_tabu_inicial: int = 10,
                             max_iteraciones: int = 1000) -> Tuple[List[int], int, int]:
    """
    Búsqueda tabú con tamaño de lista adaptativo.
    
    Args:
        problema: El problema a resolver
        estado_inicial: Estado desde donde comenzar
        tamano_tabu_inicial: Tamaño inicial de la lista tabú
        max_iteraciones: Máximo número de iteraciones
    
    Returns:
        Tupla con (mejor_estado, mejor_valor, iteraciones)
    """
    estado_actual = estado_inicial
    valor_actual = problema.evaluar(estado_actual)
    
    mejor_estado = estado_actual.copy()
    mejor_valor = valor_actual
    
    tamano_tabu = tamano_tabu_inicial
    lista_tabu = ListaTabu(tamano_tabu)
    
    iteraciones = 0
    ciclos_detectados = 0
    
    while iteraciones < max_iteraciones:
        iteraciones += 1
        
        if mejor_valor == 0:
            break
        
        vecinos = problema.obtener_vecinos(estado_actual)
        
        mejor_vecino = None
        mejor_valor_vecino = float('inf')
        mejor_movimiento = None
        
        for vecino, movimiento in vecinos:
            valor_vecino = problema.evaluar(vecino)
            es_tabu = lista_tabu.es_tabu(movimiento)
            criterio_aspiracion = valor_vecino < mejor_valor
            
            if (not es_tabu or criterio_aspiracion) and valor_vecino < mejor_valor_vecino:
                mejor_vecino = vecino
                mejor_valor_vecino = valor_vecino
                mejor_movimiento = movimiento
        
        if mejor_vecino is None:
            break
        
        estado_actual = mejor_vecino
        valor_actual = mejor_valor_vecino
        lista_tabu.agregar(mejor_movimiento)
        
        # Adaptar tamaño de lista tabú
        if valor_actual >= mejor_valor:
            ciclos_detectados += 1
            if ciclos_detectados > 10:
                tamano_tabu = min(tamano_tabu + 2, 30)
                lista_tabu = ListaTabu(tamano_tabu)
                ciclos_detectados = 0
        else:
            mejor_estado = estado_actual.copy()
            mejor_valor = valor_actual
            ciclos_detectados = 0
    
    return mejor_estado, mejor_valor, iteraciones


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Búsqueda Tabú ===\n")
    print("Problema: 8-Reinas\n")
    
    # Crear problema
    problema = ProblemaReinas(8)
    
    # Estado inicial aleatorio
    estado_inicial = problema.estado_aleatorio()
    valor_inicial = problema.evaluar(estado_inicial)
    
    print(f"Estado inicial: {estado_inicial}")
    print(f"Conflictos iniciales: {valor_inicial}\n")
    
    # Ejecutar búsqueda tabú básica
    print("--- Búsqueda Tabú Básica ---")
    estado1, valor1, iter1, historial1 = busqueda_tabu(
        problema, estado_inicial.copy(), tamano_tabu=10, max_iteraciones=1000
    )
    
    print(f"Estado final: {estado1}")
    print(f"Conflictos finales: {valor1}")
    print(f"Iteraciones: {iter1}")
    print(f"¿Solución?: {'✓ Sí' if problema.es_solucion(estado1) else '✗ No'}\n")
    
    # Ejecutar búsqueda tabú adaptativa
    print("--- Búsqueda Tabú Adaptativa ---")
    estado2, valor2, iter2 = busqueda_tabu_adaptativa(
        problema, estado_inicial.copy(), max_iteraciones=1000
    )
    
    print(f"Estado final: {estado2}")
    print(f"Conflictos finales: {valor2}")
    print(f"Iteraciones: {iter2}")
    print(f"¿Solución?: {'✓ Sí' if problema.es_solucion(estado2) else '✗ No'}\n")
    
    # Comparar con múltiples ejecuciones
    print("--- Comparación (10 ejecuciones) ---")
    exitos = 0
    total_iteraciones = 0
    
    for i in range(10):
        estado_inicial_i = problema.estado_aleatorio()
        estado_i, valor_i, iter_i, _ = busqueda_tabu(
            problema, estado_inicial_i, tamano_tabu=10, max_iteraciones=500
        )
        total_iteraciones += iter_i
        if problema.es_solucion(estado_i):
            exitos += 1
    
    print(f"Soluciones encontradas: {exitos}/10")
    print(f"Iteraciones promedio: {total_iteraciones/10:.1f}\n")
    
    print("="*50)
    print("\nCaracterísticas de la Búsqueda Tabú:")
    print("- Usa memoria (lista tabú) para evitar ciclos")
    print("- Puede aceptar movimientos que empeoran la solución")
    print("- Criterio de aspiración permite anular restricciones")
    print("- Más robusta que ascensión de colinas simple")

