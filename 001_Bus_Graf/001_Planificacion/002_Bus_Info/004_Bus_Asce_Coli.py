"""
Algoritmo 11: Búsqueda de Ascensión de Colinas (Hill Climbing)

La ascensión de colinas es un algoritmo de búsqueda local que siempre se mueve
hacia el vecino con mejor valor. Es como escalar una colina en la niebla.

Características:
- No completo: Puede quedar atrapado en máximos locales
- No óptimo: No garantiza encontrar el óptimo global
- Muy eficiente en memoria: Solo mantiene el estado actual
- Rápido pero puede fallar
"""

import random
from typing import List, Tuple, Callable, Optional
import math


class ProblemaOptimizacion:
    """Clase base para problemas de optimización"""
    
    def evaluar(self, estado) -> float:
        """Evalúa la calidad de un estado (mayor es mejor)"""
        raise NotImplementedError
    
    def obtener_vecinos(self, estado) -> List:
        """Retorna los vecinos de un estado"""
        raise NotImplementedError
    
    def estado_aleatorio(self):
        """Genera un estado aleatorio"""
        raise NotImplementedError


class ProblemaReinas(ProblemaOptimizacion):
    """Problema de las N-Reinas como optimización"""
    
    def __init__(self, n: int = 8):
        self.n = n
    
    def evaluar(self, estado: List[int]) -> float:
        """
        Evalúa un estado contando pares de reinas que NO se atacan.
        Mayor valor = mejor estado.
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
        
        # Retornar pares que NO se atacan
        max_pares = (self.n * (self.n - 1)) // 2
        return max_pares - ataques
    
    def obtener_vecinos(self, estado: List[int]) -> List[List[int]]:
        """
        Genera vecinos moviendo una reina a una nueva fila en su columna.
        """
        vecinos = []
        for col in range(self.n):
            for fila in range(self.n):
                if fila != estado[col]:
                    vecino = estado.copy()
                    vecino[col] = fila
                    vecinos.append(vecino)
        return vecinos
    
    def estado_aleatorio(self) -> List[int]:
        """Genera un estado aleatorio"""
        return [random.randint(0, self.n - 1) for _ in range(self.n)]
    
    def es_solucion(self, estado: List[int]) -> bool:
        """Verifica si es una solución válida"""
        max_pares = (self.n * (self.n - 1)) // 2
        return self.evaluar(estado) == max_pares


def ascension_colinas_simple(problema: ProblemaOptimizacion, 
                             estado_inicial,
                             max_iteraciones: int = 1000) -> Tuple[any, float, int]:
    """
    Ascensión de colinas simple (steepest ascent).
    
    Args:
        problema: El problema a resolver
        estado_inicial: Estado desde donde comenzar
        max_iteraciones: Máximo número de iteraciones
    
    Returns:
        Tupla con (mejor_estado, mejor_valor, iteraciones)
    """
    estado_actual = estado_inicial
    valor_actual = problema.evaluar(estado_actual)
    iteraciones = 0
    
    while iteraciones < max_iteraciones:
        iteraciones += 1
        
        # Obtener todos los vecinos
        vecinos = problema.obtener_vecinos(estado_actual)
        
        # Encontrar el mejor vecino
        mejor_vecino = None
        mejor_valor_vecino = valor_actual
        
        for vecino in vecinos:
            valor_vecino = problema.evaluar(vecino)
            if valor_vecino > mejor_valor_vecino:
                mejor_vecino = vecino
                mejor_valor_vecino = valor_vecino
        
        # Si no hay mejor vecino, hemos alcanzado un máximo local
        if mejor_vecino is None:
            break
        
        # Moverse al mejor vecino
        estado_actual = mejor_vecino
        valor_actual = mejor_valor_vecino
    
    return estado_actual, valor_actual, iteraciones


def ascension_colinas_estocastica(problema: ProblemaOptimizacion,
                                  estado_inicial,
                                  max_iteraciones: int = 1000) -> Tuple[any, float, int]:
    """
    Ascensión de colinas estocástica (elige aleatoriamente entre mejores vecinos).
    
    Args:
        problema: El problema a resolver
        estado_inicial: Estado desde donde comenzar
        max_iteraciones: Máximo número de iteraciones
    
    Returns:
        Tupla con (mejor_estado, mejor_valor, iteraciones)
    """
    estado_actual = estado_inicial
    valor_actual = problema.evaluar(estado_actual)
    iteraciones = 0
    
    while iteraciones < max_iteraciones:
        iteraciones += 1
        
        # Obtener vecinos que mejoran el estado actual
        vecinos_mejores = []
        for vecino in problema.obtener_vecinos(estado_actual):
            valor_vecino = problema.evaluar(vecino)
            if valor_vecino > valor_actual:
                vecinos_mejores.append((vecino, valor_vecino))
        
        # Si no hay vecinos mejores, hemos alcanzado un máximo local
        if not vecinos_mejores:
            break
        
        # Elegir aleatoriamente entre los vecinos mejores
        estado_actual, valor_actual = random.choice(vecinos_mejores)
    
    return estado_actual, valor_actual, iteraciones


def ascension_colinas_primer_mejora(problema: ProblemaOptimizacion,
                                    estado_inicial,
                                    max_iteraciones: int = 1000) -> Tuple[any, float, int]:
    """
    Ascensión de colinas de primera mejora (first-choice hill climbing).
    Se mueve al primer vecino que mejora, sin evaluar todos.
    
    Args:
        problema: El problema a resolver
        estado_inicial: Estado desde donde comenzar
        max_iteraciones: Máximo número de iteraciones
    
    Returns:
        Tupla con (mejor_estado, mejor_valor, iteraciones)
    """
    estado_actual = estado_inicial
    valor_actual = problema.evaluar(estado_actual)
    iteraciones = 0
    
    while iteraciones < max_iteraciones:
        iteraciones += 1
        
        # Obtener vecinos en orden aleatorio
        vecinos = problema.obtener_vecinos(estado_actual)
        random.shuffle(vecinos)
        
        # Buscar el primer vecino que mejora
        encontrado = False
        for vecino in vecinos:
            valor_vecino = problema.evaluar(vecino)
            if valor_vecino > valor_actual:
                estado_actual = vecino
                valor_actual = valor_vecino
                encontrado = True
                break
        
        # Si no se encontró mejora, terminar
        if not encontrado:
            break
    
    return estado_actual, valor_actual, iteraciones


def ascension_colinas_reinicio_aleatorio(problema: ProblemaOptimizacion,
                                        max_reintentos: int = 10,
                                        max_iteraciones: int = 100) -> Tuple[any, float, int]:
    """
    Ascensión de colinas con reinicio aleatorio.
    Ejecuta múltiples veces desde estados iniciales aleatorios.
    
    Args:
        problema: El problema a resolver
        max_reintentos: Número de reinicios
        max_iteraciones: Iteraciones por intento
    
    Returns:
        Tupla con (mejor_estado, mejor_valor, iteraciones_totales)
    """
    mejor_estado_global = None
    mejor_valor_global = float('-inf')
    iteraciones_totales = 0
    
    for _ in range(max_reintentos):
        estado_inicial = problema.estado_aleatorio()
        estado, valor, iteraciones = ascension_colinas_simple(
            problema, estado_inicial, max_iteraciones
        )
        iteraciones_totales += iteraciones
        
        if valor > mejor_valor_global:
            mejor_estado_global = estado
            mejor_valor_global = valor
    
    return mejor_estado_global, mejor_valor_global, iteraciones_totales


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Búsqueda de Ascensión de Colinas ===\n")
    print("Problema: 8-Reinas\n")
    
    # Crear problema de 8 reinas
    problema = ProblemaReinas(8)
    max_pares = (8 * 7) // 2
    
    # Estado inicial aleatorio
    estado_inicial = problema.estado_aleatorio()
    valor_inicial = problema.evaluar(estado_inicial)
    
    print(f"Estado inicial: {estado_inicial}")
    print(f"Valor inicial: {valor_inicial}/{max_pares} pares sin ataque\n")
    
    # 1. Ascensión simple
    print("--- Ascensión de Colinas Simple ---")
    estado1, valor1, iter1 = ascension_colinas_simple(problema, estado_inicial.copy())
    print(f"Estado final: {estado1}")
    print(f"Valor final: {valor1}/{max_pares}")
    print(f"Iteraciones: {iter1}")
    print(f"¿Solución?: {'✓ Sí' if problema.es_solucion(estado1) else '✗ No (máximo local)'}\n")
    
    # 2. Ascensión estocástica
    print("--- Ascensión de Colinas Estocástica ---")
    estado2, valor2, iter2 = ascension_colinas_estocastica(problema, estado_inicial.copy())
    print(f"Estado final: {estado2}")
    print(f"Valor final: {valor2}/{max_pares}")
    print(f"Iteraciones: {iter2}")
    print(f"¿Solución?: {'✓ Sí' if problema.es_solucion(estado2) else '✗ No (máximo local)'}\n")
    
    # 3. Ascensión con reinicio aleatorio
    print("--- Ascensión con Reinicio Aleatorio (10 intentos) ---")
    estado3, valor3, iter3 = ascension_colinas_reinicio_aleatorio(problema, 10, 100)
    print(f"Mejor estado: {estado3}")
    print(f"Mejor valor: {valor3}/{max_pares}")
    print(f"Iteraciones totales: {iter3}")
    print(f"¿Solución?: {'✓ Sí' if problema.es_solucion(estado3) else '✗ No'}\n")
    
    print("="*50)
    print("\nCaracterísticas:")
    print("- Simple: Muy fácil de implementar")
    print("- Eficiente en memoria: Solo mantiene estado actual")
    print("- Problema: Puede quedar atrapado en máximos locales")
    print("- Solución: Reinicio aleatorio aumenta probabilidad de éxito")

