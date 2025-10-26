"""
Algoritmo 14: Búsqueda de Haz Local (Local Beam Search)

La búsqueda de haz local mantiene k estados en lugar de uno solo.
En cada iteración, genera todos los sucesores de los k estados y
selecciona los k mejores para la siguiente iteración.

Características:
- Mantiene k estados simultáneamente
- No es k búsquedas independientes (comparte información)
- Puede converger rápidamente a buenas soluciones
- Versión estocástica añade aleatoriedad para diversidad
"""

import random
from typing import List, Tuple, Set


class ProblemaReinas:
    """Problema de las N-Reinas"""
    
    def __init__(self, n: int = 8):
        self.n = n
    
    def evaluar(self, estado: List[int]) -> int:
        """Cuenta conflictos (menor es mejor, 0 = solución)"""
        ataques = 0
        for i in range(len(estado)):
            for j in range(i + 1, len(estado)):
                if estado[i] == estado[j]:
                    ataques += 1
                elif abs(estado[i] - estado[j]) == abs(i - j):
                    ataques += 1
        return ataques
    
    def obtener_vecinos(self, estado: List[int]) -> List[List[int]]:
        """Genera todos los vecinos de un estado"""
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
        return self.evaluar(estado) == 0


def busqueda_haz_local(problema: ProblemaReinas,
                       k: int = 4,
                       max_iteraciones: int = 1000) -> Tuple[List[int], int, int]:
    """
    Búsqueda de haz local.
    
    Args:
        problema: El problema a resolver
        k: Número de estados a mantener
        max_iteraciones: Máximo número de iteraciones
    
    Returns:
        Tupla con (mejor_estado, mejor_valor, iteraciones)
    """
    # Generar k estados iniciales aleatorios
    estados = [problema.estado_aleatorio() for _ in range(k)]
    
    mejor_estado = None
    mejor_valor = float('inf')
    iteraciones = 0
    
    while iteraciones < max_iteraciones:
        iteraciones += 1
        
        # Evaluar estados actuales y actualizar mejor
        for estado in estados:
            valor = problema.evaluar(estado)
            if valor < mejor_valor:
                mejor_estado = estado.copy()
                mejor_valor = valor
        
        # Si encontramos solución, terminar
        if mejor_valor == 0:
            break
        
        # Generar todos los sucesores de los k estados
        sucesores = []
        for estado in estados:
            vecinos = problema.obtener_vecinos(estado)
            sucesores.extend(vecinos)
        
        # Evaluar todos los sucesores
        sucesores_evaluados = [(s, problema.evaluar(s)) for s in sucesores]
        
        # Ordenar por valor (menor es mejor)
        sucesores_evaluados.sort(key=lambda x: x[1])
        
        # Seleccionar los k mejores
        estados = [s for s, v in sucesores_evaluados[:k]]
        
        # Si no hay mejora, terminar
        if all(problema.evaluar(s) >= mejor_valor for s in estados):
            break
    
    return mejor_estado, mejor_valor, iteraciones


def busqueda_haz_local_estocastica(problema: ProblemaReinas,
                                   k: int = 4,
                                   max_iteraciones: int = 1000) -> Tuple[List[int], int, int]:
    """
    Búsqueda de haz local estocástica.
    Selecciona estados con probabilidad proporcional a su calidad.
    
    Args:
        problema: El problema a resolver
        k: Número de estados a mantener
        max_iteraciones: Máximo número de iteraciones
    
    Returns:
        Tupla con (mejor_estado, mejor_valor, iteraciones)
    """
    # Generar k estados iniciales aleatorios
    estados = [problema.estado_aleatorio() for _ in range(k)]
    
    mejor_estado = None
    mejor_valor = float('inf')
    iteraciones = 0
    
    while iteraciones < max_iteraciones:
        iteraciones += 1
        
        # Evaluar estados actuales
        for estado in estados:
            valor = problema.evaluar(estado)
            if valor < mejor_valor:
                mejor_estado = estado.copy()
                mejor_valor = valor
        
        if mejor_valor == 0:
            break
        
        # Generar todos los sucesores
        sucesores = []
        for estado in estados:
            vecinos = problema.obtener_vecinos(estado)
            sucesores.extend(vecinos)
        
        # Evaluar sucesores
        sucesores_evaluados = [(s, problema.evaluar(s)) for s in sucesores]
        
        # Calcular pesos (invertir valores para que menor sea mejor)
        # Usamos max_valor - valor para que mejores estados tengan mayor peso
        max_valor = max(v for s, v in sucesores_evaluados)
        pesos = [max_valor - v + 1 for s, v in sucesores_evaluados]
        suma_pesos = sum(pesos)
        
        # Seleccionar k estados estocásticamente
        if suma_pesos > 0:
            probabilidades = [p / suma_pesos for p in pesos]
            indices = random.choices(range(len(sucesores)), weights=probabilidades, k=k)
            estados = [sucesores[i] for i in indices]
        else:
            # Si todos tienen el mismo valor, seleccionar aleatoriamente
            estados = random.sample(sucesores, min(k, len(sucesores)))
    
    return mejor_estado, mejor_valor, iteraciones


def busqueda_haz_local_paralela(problema: ProblemaReinas,
                                k: int = 4,
                                max_iteraciones: int = 1000) -> Tuple[List[int], int, int]:
    """
    Búsqueda de haz local con reinicio cuando todos convergen.
    
    Args:
        problema: El problema a resolver
        k: Número de estados a mantener
        max_iteraciones: Máximo número de iteraciones
    
    Returns:
        Tupla con (mejor_estado, mejor_valor, iteraciones)
    """
    estados = [problema.estado_aleatorio() for _ in range(k)]
    
    mejor_estado = None
    mejor_valor = float('inf')
    iteraciones = 0
    iteraciones_sin_mejora = 0
    
    while iteraciones < max_iteraciones:
        iteraciones += 1
        
        # Evaluar estados actuales
        valores_actuales = []
        for estado in estados:
            valor = problema.evaluar(estado)
            valores_actuales.append(valor)
            if valor < mejor_valor:
                mejor_estado = estado.copy()
                mejor_valor = valor
                iteraciones_sin_mejora = 0
        
        if mejor_valor == 0:
            break
        
        iteraciones_sin_mejora += 1
        
        # Si no hay mejora por mucho tiempo, reiniciar algunos estados
        if iteraciones_sin_mejora > 50:
            # Mantener el mejor, reiniciar el resto
            mejor_idx = valores_actuales.index(min(valores_actuales))
            for i in range(k):
                if i != mejor_idx:
                    estados[i] = problema.estado_aleatorio()
            iteraciones_sin_mejora = 0
            continue
        
        # Generar sucesores
        sucesores = []
        for estado in estados:
            vecinos = problema.obtener_vecinos(estado)
            sucesores.extend(vecinos)
        
        # Evaluar y seleccionar mejores
        sucesores_evaluados = [(s, problema.evaluar(s)) for s in sucesores]
        sucesores_evaluados.sort(key=lambda x: x[1])
        estados = [s for s, v in sucesores_evaluados[:k]]
    
    return mejor_estado, mejor_valor, iteraciones


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Búsqueda de Haz Local (Local Beam Search) ===\n")
    print("Problema: 8-Reinas\n")
    
    problema = ProblemaReinas(8)
    
    # Búsqueda de haz local básica
    print("--- Búsqueda de Haz Local Básica (k=4) ---")
    estado1, valor1, iter1 = busqueda_haz_local(problema, k=4, max_iteraciones=500)
    print(f"Estado final: {estado1}")
    print(f"Conflictos: {valor1}")
    print(f"Iteraciones: {iter1}")
    print(f"¿Solución?: {'✓ Sí' if problema.es_solucion(estado1) else '✗ No'}\n")
    
    # Búsqueda de haz local estocástica
    print("--- Búsqueda de Haz Local Estocástica (k=4) ---")
    estado2, valor2, iter2 = busqueda_haz_local_estocastica(problema, k=4, max_iteraciones=500)
    print(f"Estado final: {estado2}")
    print(f"Conflictos: {valor2}")
    print(f"Iteraciones: {iter2}")
    print(f"¿Solución?: {'✓ Sí' if problema.es_solucion(estado2) else '✗ No'}\n")
    
    # Búsqueda de haz local con reinicio
    print("--- Búsqueda de Haz Local con Reinicio (k=4) ---")
    estado3, valor3, iter3 = busqueda_haz_local_paralela(problema, k=4, max_iteraciones=500)
    print(f"Estado final: {estado3}")
    print(f"Conflictos: {valor3}")
    print(f"Iteraciones: {iter3}")
    print(f"¿Solución?: {'✓ Sí' if problema.es_solucion(estado3) else '✗ No'}\n")
    
    # Comparar diferentes valores de k
    print("--- Comparación de diferentes valores de k ---")
    for k in [2, 4, 8, 16]:
        exitos = 0
        for _ in range(5):
            _, valor, _ = busqueda_haz_local(problema, k=k, max_iteraciones=500)
            if valor == 0:
                exitos += 1
        print(f"k={k:2d}: {exitos}/5 soluciones encontradas")
    
    print("\n" + "="*50)
    print("\nCaracterísticas:")
    print("- Mantiene k estados simultáneamente")
    print("- Comparte información entre estados (no son independientes)")
    print("- Versión estocástica añade diversidad")
    print("- Más robusto que búsqueda local simple")
    print("- k más grande: Más exploración pero más costoso")

