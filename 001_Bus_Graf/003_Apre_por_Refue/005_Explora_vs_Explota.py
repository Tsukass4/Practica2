"""
Algoritmo 36: Exploración vs. Explotación

El dilema exploración-explotación es fundamental en aprendizaje por refuerzo:
- Exploración: Probar acciones nuevas para descubrir mejores opciones
- Explotación: Usar el conocimiento actual para maximizar recompensa

Estrategias:
- Epsilon-greedy: Explorar con probabilidad ε
- Softmax/Boltzmann: Probabilidad proporcional a valores estimados
- UCB (Upper Confidence Bound): Balance basado en incertidumbre
- Thompson Sampling: Muestreo bayesiano
"""

from typing import List, Tuple, Dict
import random
import math
import numpy as np
from collections import defaultdict


class BanditMultibrazo:
    """
    Problema del bandido multibrazo (multi-armed bandit):
    Múltiples máquinas tragamonedas con recompensas desconocidas.
    """
    
    def __init__(self, num_brazos: int, medias_reales: List[float] = None):
        """
        Args:
            num_brazos: Número de brazos (acciones)
            medias_reales: Recompensas medias reales de cada brazo
        """
        self.num_brazos = num_brazos
        
        if medias_reales is None:
            # Generar medias aleatorias
            self.medias_reales = [random.uniform(0, 1) for _ in range(num_brazos)]
        else:
            self.medias_reales = medias_reales
        
        self.mejor_brazo = max(range(num_brazos), key=lambda i: self.medias_reales[i])
    
    def tirar(self, brazo: int) -> float:
        """Tira de un brazo y obtiene recompensa (con ruido gaussiano)"""
        return self.medias_reales[brazo] + random.gauss(0, 0.1)


class EstrategiaEpsilonGreedy:
    """Estrategia epsilon-greedy"""
    
    def __init__(self, num_brazos: int, epsilon: float = 0.1):
        self.num_brazos = num_brazos
        self.epsilon = epsilon
        self.valores_estimados = [0.0] * num_brazos
        self.conteos = [0] * num_brazos
    
    def elegir_brazo(self) -> int:
        """Elige brazo usando epsilon-greedy"""
        if random.random() < self.epsilon:
            # Exploración
            return random.randint(0, self.num_brazos - 1)
        else:
            # Explotación
            return max(range(self.num_brazos), key=lambda i: self.valores_estimados[i])
    
    def actualizar(self, brazo: int, recompensa: float):
        """Actualiza estimación del valor del brazo"""
        self.conteos[brazo] += 1
        n = self.conteos[brazo]
        
        # Promedio incremental
        valor_anterior = self.valores_estimados[brazo]
        self.valores_estimados[brazo] = valor_anterior + (recompensa - valor_anterior) / n


class EstrategiaSoftmax:
    """Estrategia Softmax (Boltzmann)"""
    
    def __init__(self, num_brazos: int, temperatura: float = 0.1):
        self.num_brazos = num_brazos
        self.temperatura = temperatura
        self.valores_estimados = [0.0] * num_brazos
        self.conteos = [0] * num_brazos
    
    def elegir_brazo(self) -> int:
        """Elige brazo usando distribución softmax"""
        # Calcular probabilidades softmax
        exp_valores = [math.exp(v / self.temperatura) for v in self.valores_estimados]
        suma = sum(exp_valores)
        probabilidades = [e / suma for e in exp_valores]
        
        # Muestrear
        return np.random.choice(self.num_brazos, p=probabilidades)
    
    def actualizar(self, brazo: int, recompensa: float):
        """Actualiza estimación"""
        self.conteos[brazo] += 1
        n = self.conteos[brazo]
        valor_anterior = self.valores_estimados[brazo]
        self.valores_estimados[brazo] = valor_anterior + (recompensa - valor_anterior) / n


class EstrategiaUCB:
    """Upper Confidence Bound (UCB1)"""
    
    def __init__(self, num_brazos: int, c: float = 2.0):
        self.num_brazos = num_brazos
        self.c = c  # Parámetro de exploración
        self.valores_estimados = [0.0] * num_brazos
        self.conteos = [0] * num_brazos
        self.t = 0  # Tiempo total
    
    def elegir_brazo(self) -> int:
        """Elige brazo usando UCB"""
        # Primero, asegurar que todos los brazos se prueban al menos una vez
        for i in range(self.num_brazos):
            if self.conteos[i] == 0:
                return i
        
        # Calcular UCB para cada brazo
        ucb_valores = []
        for i in range(self.num_brazos):
            valor_medio = self.valores_estimados[i]
            bonificacion_exploracion = self.c * math.sqrt(math.log(self.t) / self.conteos[i])
            ucb_valores.append(valor_medio + bonificacion_exploracion)
        
        return max(range(self.num_brazos), key=lambda i: ucb_valores[i])
    
    def actualizar(self, brazo: int, recompensa: float):
        """Actualiza estimación"""
        self.conteos[brazo] += 1
        self.t += 1
        n = self.conteos[brazo]
        valor_anterior = self.valores_estimados[brazo]
        self.valores_estimados[brazo] = valor_anterior + (recompensa - valor_anterior) / n


class EstrategiaThompsonSampling:
    """Thompson Sampling (Bayesiano)"""
    
    def __init__(self, num_brazos: int):
        self.num_brazos = num_brazos
        # Prior: Beta(1, 1) = Uniforme
        self.alpha = [1.0] * num_brazos  # Éxitos + 1
        self.beta = [1.0] * num_brazos   # Fracasos + 1
    
    def elegir_brazo(self) -> int:
        """Elige brazo muestreando de la distribución posterior"""
        muestras = [random.betavariate(self.alpha[i], self.beta[i]) 
                   for i in range(self.num_brazos)]
        return max(range(self.num_brazos), key=lambda i: muestras[i])
    
    def actualizar(self, brazo: int, recompensa: float):
        """Actualiza distribución posterior (asumiendo recompensa binaria)"""
        # Para recompensas continuas, convertir a binaria
        exito = 1 if recompensa > 0.5 else 0
        
        if exito:
            self.alpha[brazo] += 1
        else:
            self.beta[brazo] += 1


def comparar_estrategias():
    """Compara diferentes estrategias de exploración-explotación"""
    print("=== Exploración vs. Explotación ===\n")
    
    # Crear bandido con 5 brazos
    medias_reales = [0.1, 0.3, 0.7, 0.4, 0.5]
    num_pasos = 1000
    num_experimentos = 100
    
    print(f"Bandido multibrazo con {len(medias_reales)} brazos")
    print(f"Recompensas medias reales: {medias_reales}")
    print(f"Mejor brazo: {medias_reales.index(max(medias_reales))} (recompensa = {max(medias_reales)})")
    print(f"\nSimulando {num_experimentos} experimentos de {num_pasos} pasos cada uno...\n")
    
    estrategias = [
        ("Epsilon-Greedy (ε=0.1)", lambda: EstrategiaEpsilonGreedy(len(medias_reales), 0.1)),
        ("Epsilon-Greedy (ε=0.01)", lambda: EstrategiaEpsilonGreedy(len(medias_reales), 0.01)),
        ("Softmax (τ=0.1)", lambda: EstrategiaSoftmax(len(medias_reales), 0.1)),
        ("UCB (c=2)", lambda: EstrategiaUCB(len(medias_reales), 2.0)),
        ("Thompson Sampling", lambda: EstrategiaThompsonSampling(len(medias_reales))),
    ]
    
    resultados = {}
    
    for nombre, crear_estrategia in estrategias:
        recompensas_totales = []
        selecciones_optimas = []
        
        for _ in range(num_experimentos):
            bandido = BanditMultibrazo(len(medias_reales), medias_reales)
            estrategia = crear_estrategia()
            
            recompensa_total = 0.0
            selecciones_optimas_exp = 0
            
            for paso in range(num_pasos):
                brazo = estrategia.elegir_brazo()
                recompensa = bandido.tirar(brazo)
                estrategia.actualizar(brazo, recompensa)
                
                recompensa_total += recompensa
                if brazo == bandido.mejor_brazo:
                    selecciones_optimas_exp += 1
            
            recompensas_totales.append(recompensa_total)
            selecciones_optimas.append(selecciones_optimas_exp / num_pasos * 100)
        
        resultados[nombre] = {
            'recompensa_media': np.mean(recompensas_totales),
            'recompensa_std': np.std(recompensas_totales),
            'optimalidad': np.mean(selecciones_optimas)
        }
    
    # Mostrar resultados
    print("Resultados (promedio de {} experimentos):".format(num_experimentos))
    print("="*80)
    print(f"{'Estrategia':<30} {'Recompensa Total':<20} {'% Óptimo':<15}")
    print("="*80)
    
    for nombre in resultados:
        r = resultados[nombre]
        print(f"{nombre:<30} {r['recompensa_media']:>8.2f} ± {r['recompensa_std']:>5.2f}      {r['optimalidad']:>6.1f}%")
    
    print("\n" + "="*80)
    
    # Análisis
    print("\nAnálisis:")
    mejor_estrategia = max(resultados.items(), key=lambda x: x[1]['recompensa_media'])
    print(f"Mejor estrategia: {mejor_estrategia[0]}")
    print(f"Recompensa promedio: {mejor_estrategia[1]['recompensa_media']:.2f}")


def ejemplo_decaimiento_epsilon():
    """Demuestra el decaimiento de epsilon en el tiempo"""
    print("\n\n=== Decaimiento de Epsilon ===\n")
    
    print("Estrategias de decaimiento:")
    print("1. Lineal: ε(t) = ε_0 * (1 - t/T)")
    print("2. Exponencial: ε(t) = ε_0 * decay^t")
    print("3. Inverso: ε(t) = ε_0 / (1 + t/k)")
    print()
    
    epsilon_inicial = 1.0
    pasos = [0, 100, 200, 500, 1000]
    
    print(f"{'Paso':<10} {'Lineal':<15} {'Exponencial':<15} {'Inverso':<15}")
    print("-" * 55)
    
    for t in pasos:
        # Lineal
        eps_lineal = max(0.01, epsilon_inicial * (1 - t / 1000))
        
        # Exponencial
        eps_exp = max(0.01, epsilon_inicial * (0.995 ** t))
        
        # Inverso
        eps_inv = epsilon_inicial / (1 + t / 100)
        
        print(f"{t:<10} {eps_lineal:<15.3f} {eps_exp:<15.3f} {eps_inv:<15.3f}")
    
    print("\nRecomendaciones:")
    print("- Inicio: Alta exploración (ε ≈ 1.0)")
    print("- Final: Baja exploración (ε ≈ 0.01-0.1)")
    print("- Exponencial: Buena para la mayoría de casos")
    print("- UCB/Thompson: No requieren decaimiento manual")


# Ejecutar ejemplos
if __name__ == "__main__":
    comparar_estrategias()
    ejemplo_decaimiento_epsilon()
    
    print("\n" + "="*70)
    print("\nResumen de Estrategias:")
    print("\nEpsilon-Greedy:")
    print("  + Simple de implementar")
    print("  + Funciona bien en práctica")
    print("  - Explora uniformemente (no considera incertidumbre)")
    print("\nSoftmax:")
    print("  + Explora proporcionalmente a valores")
    print("  - Sensible al parámetro de temperatura")
    print("\nUCB:")
    print("  + Explora basado en incertidumbre")
    print("  + Garantías teóricas de rendimiento")
    print("  - Requiere conteos de visitas")
    print("\nThompson Sampling:")
    print("  + Enfoque bayesiano elegante")
    print("  + Excelente rendimiento empírico")
    print("  + Adapta exploración automáticamente")

