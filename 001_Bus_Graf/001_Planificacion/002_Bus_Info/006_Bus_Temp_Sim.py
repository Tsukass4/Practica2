"""
Algoritmo 13: Búsqueda de Temple Simulado (Simulated Annealing)

El temple simulado es una metaheurística inspirada en el proceso metalúrgico
de recocido. Permite movimientos que empeoran la solución con probabilidad
decreciente, lo que ayuda a escapar de óptimos locales.

Características:
- Probabilístico: Acepta movimientos malos con cierta probabilidad
- Temperatura: Controla la probabilidad de aceptación
- Enfriamiento: La temperatura disminuye gradualmente
- Balance: Exploración (alta T) vs. explotación (baja T)
"""

import random
import math
from typing import List, Tuple, Callable


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
                if estado[i] == estado[j]:
                    ataques += 1
                elif abs(estado[i] - estado[j]) == abs(i - j):
                    ataques += 1
        return ataques
    
    def obtener_vecino_aleatorio(self, estado: List[int]) -> List[int]:
        """Genera un vecino aleatorio moviendo una reina"""
        vecino = estado.copy()
        col = random.randint(0, self.n - 1)
        fila = random.randint(0, self.n - 1)
        vecino[col] = fila
        return vecino
    
    def estado_aleatorio(self) -> List[int]:
        """Genera un estado aleatorio"""
        return [random.randint(0, self.n - 1) for _ in range(self.n)]
    
    def es_solucion(self, estado: List[int]) -> bool:
        """Verifica si es una solución válida"""
        return self.evaluar(estado) == 0


def probabilidad_aceptacion(delta_e: float, temperatura: float) -> float:
    """
    Calcula la probabilidad de aceptar un movimiento que empeora la solución.
    
    Args:
        delta_e: Cambio en energía (valor_nuevo - valor_actual)
        temperatura: Temperatura actual
    
    Returns:
        Probabilidad de aceptación (0 a 1)
    """
    if delta_e < 0:  # Mejora
        return 1.0
    if temperatura == 0:
        return 0.0
    return math.exp(-delta_e / temperatura)


def enfriamiento_lineal(temperatura: float, tasa: float) -> float:
    """Enfriamiento lineal: T = T - tasa"""
    return max(0, temperatura - tasa)


def enfriamiento_exponencial(temperatura: float, alpha: float) -> float:
    """Enfriamiento exponencial: T = T * alpha"""
    return temperatura * alpha


def enfriamiento_logaritmico(temperatura: float, iteracion: int) -> float:
    """Enfriamiento logarítmico: T = T0 / log(1 + iteracion)"""
    return temperatura / math.log(2 + iteracion)


def temple_simulado(problema: ProblemaReinas,
                   estado_inicial: List[int],
                   temperatura_inicial: float = 100.0,
                   temperatura_minima: float = 0.01,
                   alpha: float = 0.95,
                   iteraciones_por_temperatura: int = 100,
                   max_iteraciones: int = 10000) -> Tuple[List[int], int, int, List[Tuple[int, float]]]:
    """
    Implementación del temple simulado con enfriamiento exponencial.
    
    Args:
        problema: El problema a resolver
        estado_inicial: Estado desde donde comenzar
        temperatura_inicial: Temperatura inicial
        temperatura_minima: Temperatura mínima (criterio de parada)
        alpha: Factor de enfriamiento (0 < alpha < 1)
        iteraciones_por_temperatura: Iteraciones antes de enfriar
        max_iteraciones: Máximo número total de iteraciones
    
    Returns:
        Tupla con (mejor_estado, mejor_valor, iteraciones, historial)
    """
    # Inicializar
    estado_actual = estado_inicial
    valor_actual = problema.evaluar(estado_actual)
    
    mejor_estado = estado_actual.copy()
    mejor_valor = valor_actual
    
    temperatura = temperatura_inicial
    iteraciones = 0
    historial = [(valor_actual, temperatura)]
    
    while temperatura > temperatura_minima and iteraciones < max_iteraciones:
        for _ in range(iteraciones_por_temperatura):
            iteraciones += 1
            
            # Si encontramos la solución óptima, terminar
            if mejor_valor == 0:
                return mejor_estado, mejor_valor, iteraciones, historial
            
            # Generar vecino aleatorio
            vecino = problema.obtener_vecino_aleatorio(estado_actual)
            valor_vecino = problema.evaluar(vecino)
            
            # Calcular cambio en energía (en problemas de minimización)
            delta_e = valor_vecino - valor_actual
            
            # Decidir si aceptar el movimiento
            if delta_e < 0 or random.random() < probabilidad_aceptacion(delta_e, temperatura):
                estado_actual = vecino
                valor_actual = valor_vecino
                
                # Actualizar mejor solución
                if valor_actual < mejor_valor:
                    mejor_estado = estado_actual.copy()
                    mejor_valor = valor_actual
            
            historial.append((valor_actual, temperatura))
        
        # Enfriar
        temperatura = enfriamiento_exponencial(temperatura, alpha)
    
    return mejor_estado, mejor_valor, iteraciones, historial


def temple_simulado_adaptativo(problema: ProblemaReinas,
                              estado_inicial: List[int],
                              temperatura_inicial: float = 100.0,
                              max_iteraciones: int = 10000) -> Tuple[List[int], int, int]:
    """
    Temple simulado con enfriamiento adaptativo.
    Ajusta la tasa de enfriamiento según el progreso.
    
    Args:
        problema: El problema a resolver
        estado_inicial: Estado desde donde comenzar
        temperatura_inicial: Temperatura inicial
        max_iteraciones: Máximo número de iteraciones
    
    Returns:
        Tupla con (mejor_estado, mejor_valor, iteraciones)
    """
    estado_actual = estado_inicial
    valor_actual = problema.evaluar(estado_actual)
    
    mejor_estado = estado_actual.copy()
    mejor_valor = valor_actual
    
    temperatura = temperatura_inicial
    iteraciones = 0
    aceptaciones = 0
    intentos = 0
    
    while iteraciones < max_iteraciones and temperatura > 0.01:
        iteraciones += 1
        
        if mejor_valor == 0:
            break
        
        vecino = problema.obtener_vecino_aleatorio(estado_actual)
        valor_vecino = problema.evaluar(vecino)
        delta_e = valor_vecino - valor_actual
        
        intentos += 1
        if delta_e < 0 or random.random() < probabilidad_aceptacion(delta_e, temperatura):
            estado_actual = vecino
            valor_actual = valor_vecino
            aceptaciones += 1
            
            if valor_actual < mejor_valor:
                mejor_estado = estado_actual.copy()
                mejor_valor = valor_actual
        
        # Ajustar temperatura adaptativamente cada 100 iteraciones
        if iteraciones % 100 == 0:
            tasa_aceptacion = aceptaciones / intentos if intentos > 0 else 0
            
            # Si aceptamos mucho, enfriar más rápido
            if tasa_aceptacion > 0.5:
                temperatura *= 0.90
            # Si aceptamos poco, enfriar más lento
            elif tasa_aceptacion < 0.2:
                temperatura *= 0.98
            else:
                temperatura *= 0.95
            
            aceptaciones = 0
            intentos = 0
    
    return mejor_estado, mejor_valor, iteraciones


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Búsqueda de Temple Simulado (Simulated Annealing) ===\n")
    print("Problema: 8-Reinas\n")
    
    # Crear problema
    problema = ProblemaReinas(8)
    
    # Estado inicial aleatorio
    estado_inicial = problema.estado_aleatorio()
    valor_inicial = problema.evaluar(estado_inicial)
    
    print(f"Estado inicial: {estado_inicial}")
    print(f"Conflictos iniciales: {valor_inicial}\n")
    
    # Ejecutar temple simulado
    print("--- Temple Simulado Básico ---")
    estado1, valor1, iter1, historial1 = temple_simulado(
        problema, estado_inicial.copy(),
        temperatura_inicial=100.0,
        alpha=0.95,
        iteraciones_por_temperatura=100
    )
    
    print(f"Estado final: {estado1}")
    print(f"Conflictos finales: {valor1}")
    print(f"Iteraciones: {iter1}")
    print(f"¿Solución?: {'✓ Sí' if problema.es_solucion(estado1) else '✗ No'}\n")
    
    # Ejecutar temple simulado adaptativo
    print("--- Temple Simulado Adaptativo ---")
    estado2, valor2, iter2 = temple_simulado_adaptativo(
        problema, estado_inicial.copy(),
        temperatura_inicial=100.0
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
        estado_i, valor_i, iter_i, _ = temple_simulado(
            problema, estado_inicial_i,
            temperatura_inicial=100.0,
            alpha=0.95
        )
        total_iteraciones += iter_i
        if problema.es_solucion(estado_i):
            exitos += 1
    
    print(f"Soluciones encontradas: {exitos}/10")
    print(f"Iteraciones promedio: {total_iteraciones/10:.1f}\n")
    
    print("="*50)
    print("\nCaracterísticas del Temple Simulado:")
    print("- Inspirado en el proceso de recocido metalúrgico")
    print("- Acepta movimientos malos con probabilidad P = e^(-ΔE/T)")
    print("- Alta temperatura: Más exploración")
    print("- Baja temperatura: Más explotación")
    print("- Garantiza convergencia con enfriamiento logarítmico")
    print("\nParámetros clave:")
    print("- Temperatura inicial: Debe permitir exploración amplia")
    print("- Tasa de enfriamiento: Balance entre calidad y tiempo")
    print("- Iteraciones por temperatura: Suficientes para equilibrar")

