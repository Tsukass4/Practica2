"""
Algoritmo 37: Búsqueda de la Política (Policy Search)

La búsqueda de políticas optimiza directamente la política parametrizada
en lugar de aprender una función de valor. Es especialmente útil para:
- Espacios de acción continuos
- Políticas estocásticas
- Problemas de alta dimensionalidad

Métodos:
- REINFORCE (Policy Gradient)
- Actor-Critic
- Algoritmos genéticos para políticas
- Hill Climbing en espacio de políticas
"""

from typing import List, Tuple, Dict, Callable
import random
import math
import numpy as np
from collections import defaultdict


class PoliticaParametrizada:
    """Política parametrizada simple (lineal)"""
    
    def __init__(self, dim_estado: int, num_acciones: int):
        """
        Args:
            dim_estado: Dimensión del vector de estado
            num_acciones: Número de acciones discretas
        """
        self.dim_estado = dim_estado
        self.num_acciones = num_acciones
        
        # Parámetros: matriz de pesos
        self.theta = np.random.randn(dim_estado, num_acciones) * 0.01
    
    def probabilidades_acciones(self, estado: np.ndarray) -> np.ndarray:
        """Calcula probabilidades de acciones usando softmax"""
        logits = estado @ self.theta
        exp_logits = np.exp(logits - np.max(logits))  # Estabilidad numérica
        return exp_logits / np.sum(exp_logits)
    
    def elegir_accion(self, estado: np.ndarray) -> int:
        """Muestrea acción de la distribución de la política"""
        probs = self.probabilidades_acciones(estado)
        return np.random.choice(self.num_acciones, p=probs)
    
    def log_prob_accion(self, estado: np.ndarray, accion: int) -> float:
        """Calcula log-probabilidad de una acción"""
        probs = self.probabilidades_acciones(estado)
        return np.log(probs[accion] + 1e-10)


class REINFORCE:
    """
    Algoritmo REINFORCE (Williams, 1992)
    
    Actualización de gradiente de política:
    θ ← θ + α ∇_θ log π(a|s,θ) G_t
    
    donde G_t es el retorno desde el tiempo t.
    """
    
    def __init__(self, dim_estado: int, num_acciones: int, 
                 alpha: float = 0.01, gamma: float = 0.99):
        self.politica = PoliticaParametrizada(dim_estado, num_acciones)
        self.alpha = alpha
        self.gamma = gamma
    
    def entrenar_episodio(self, entorno, max_pasos: int = 500) -> float:
        """
        Entrena un episodio usando REINFORCE.
        
        Returns:
            Retorno total del episodio
        """
        # Recolectar trayectoria
        estados = []
        acciones = []
        recompensas = []
        
        estado = entorno.reiniciar()
        
        for paso in range(max_pasos):
            estados.append(estado)
            accion = self.politica.elegir_accion(estado)
            acciones.append(accion)
            
            estado, recompensa, terminal = entorno.paso(accion)
            recompensas.append(recompensa)
            
            if terminal:
                break
        
        # Calcular retornos
        T = len(recompensas)
        retornos = np.zeros(T)
        G = 0
        
        for t in range(T - 1, -1, -1):
            G = recompensas[t] + self.gamma * G
            retornos[t] = G
        
        # Normalizar retornos (reduce varianza)
        if len(retornos) > 1:
            retornos = (retornos - np.mean(retornos)) / (np.std(retornos) + 1e-10)
        
        # Actualizar parámetros
        for t in range(T):
            estado_t = estados[t]
            accion_t = acciones[t]
            G_t = retornos[t]
            
            # Calcular gradiente
            probs = self.politica.probabilidades_acciones(estado_t)
            
            # Gradiente de log π(a|s)
            grad_log = np.zeros((self.politica.dim_estado, self.politica.num_acciones))
            
            for a in range(self.politica.num_acciones):
                if a == accion_t:
                    grad_log[:, a] = estado_t * (1 - probs[a])
                else:
                    grad_log[:, a] = -estado_t * probs[a]
            
            # Actualización de gradiente de política
            self.politica.theta += self.alpha * grad_log * G_t
        
        return sum(recompensas)


class HillClimbingPolitica:
    """Hill Climbing en espacio de políticas"""
    
    def __init__(self, dim_estado: int, num_acciones: int, 
                 ruido: float = 0.1):
        self.politica = PoliticaParametrizada(dim_estado, num_acciones)
        self.ruido = ruido
        self.mejor_theta = self.politica.theta.copy()
        self.mejor_rendimiento = float('-inf')
    
    def evaluar_politica(self, entorno, num_episodios: int = 5) -> float:
        """Evalúa el rendimiento promedio de la política actual"""
        retornos = []
        
        for _ in range(num_episodios):
            estado = entorno.reiniciar()
            retorno = 0.0
            
            for paso in range(500):
                accion = self.politica.elegir_accion(estado)
                estado, recompensa, terminal = entorno.paso(accion)
                retorno += recompensa
                
                if terminal:
                    break
            
            retornos.append(retorno)
        
        return np.mean(retornos)
    
    def entrenar_iteracion(self, entorno) -> float:
        """Una iteración de hill climbing"""
        # Generar política candidata (perturbación aleatoria)
        theta_candidato = self.mejor_theta + np.random.randn(*self.mejor_theta.shape) * self.ruido
        
        # Evaluar candidato
        self.politica.theta = theta_candidato
        rendimiento_candidato = self.evaluar_politica(entorno)
        
        # Actualizar si es mejor
        if rendimiento_candidato > self.mejor_rendimiento:
            self.mejor_theta = theta_candidato
            self.mejor_rendimiento = rendimiento_candidato
        else:
            # Volver a la mejor política
            self.politica.theta = self.mejor_theta
        
        return self.mejor_rendimiento


class EntornoCartPoleSimplificado:
    """Versión simplificada de CartPole para demostración"""
    
    def __init__(self):
        self.dim_estado = 4
        self.num_acciones = 2  # Izquierda, Derecha
        self.estado = None
    
    def reiniciar(self) -> np.ndarray:
        """Reinicia el entorno"""
        # Estado: [posición, velocidad, ángulo, velocidad_angular]
        self.estado = np.random.randn(self.dim_estado) * 0.1
        self.pasos = 0
        return self.estado.copy()
    
    def paso(self, accion: int) -> Tuple[np.ndarray, float, bool]:
        """
        Ejecuta un paso.
        
        Returns:
            (nuevo_estado, recompensa, terminal)
        """
        # Dinámica simplificada
        fuerza = 1.0 if accion == 1 else -1.0
        
        pos, vel, ang, vel_ang = self.estado
        
        # Actualización simplificada
        vel += fuerza * 0.1 + ang * 0.5
        pos += vel * 0.1
        vel_ang += ang * 0.5 + fuerza * 0.05
        ang += vel_ang * 0.1
        
        self.estado = np.array([pos, vel, ang, vel_ang])
        self.pasos += 1
        
        # Terminal si se cae o sale de límites
        terminal = abs(ang) > 0.5 or abs(pos) > 2.0 or self.pasos >= 200
        
        # Recompensa: 1 por cada paso que sobrevive
        recompensa = 1.0 if not terminal else 0.0
        
        return self.estado.copy(), recompensa, terminal


# Ejemplo de uso
def ejemplo_reinforce():
    """Entrena un agente usando REINFORCE"""
    print("=== Búsqueda de la Política: REINFORCE ===\n")
    
    entorno = EntornoCartPoleSimplificado()
    agente = REINFORCE(
        dim_estado=entorno.dim_estado,
        num_acciones=entorno.num_acciones,
        alpha=0.01,
        gamma=0.99
    )
    
    print("Entrenando agente REINFORCE...")
    num_episodios = 200
    ventana_promedio = 20
    
    retornos = []
    
    for episodio in range(num_episodios):
        retorno = agente.entrenar_episodio(entorno)
        retornos.append(retorno)
        
        if (episodio + 1) % 50 == 0:
            promedio_reciente = np.mean(retornos[-ventana_promedio:])
            print(f"Episodio {episodio + 1}: Retorno promedio (últimos {ventana_promedio}) = {promedio_reciente:.2f}")
    
    print(f"\nRetorno final promedio: {np.mean(retornos[-ventana_promedio:]):.2f}")


def ejemplo_hill_climbing():
    """Entrena un agente usando Hill Climbing"""
    print("\n\n=== Búsqueda de la Política: Hill Climbing ===\n")
    
    entorno = EntornoCartPoleSimplificado()
    agente = HillClimbingPolitica(
        dim_estado=entorno.dim_estado,
        num_acciones=entorno.num_acciones,
        ruido=0.1
    )
    
    print("Entrenando agente Hill Climbing...")
    num_iteraciones = 100
    
    for iteracion in range(num_iteraciones):
        rendimiento = agente.entrenar_iteracion(entorno)
        
        if (iteracion + 1) % 20 == 0:
            print(f"Iteración {iteracion + 1}: Mejor rendimiento = {rendimiento:.2f}")
    
    print(f"\nRendimiento final: {agente.mejor_rendimiento:.2f}")


def comparacion_metodos():
    """Compara diferentes métodos de búsqueda de políticas"""
    print("\n\n=== Comparación de Métodos ===\n")
    
    print("Métodos de Búsqueda de Políticas:")
    print("\n1. REINFORCE (Policy Gradient):")
    print("   + Gradiente verdadero de la política")
    print("   + Funciona con acciones continuas")
    print("   - Alta varianza")
    print("   - Requiere episodios completos")
    
    print("\n2. Actor-Critic:")
    print("   + Menor varianza que REINFORCE")
    print("   + Aprendizaje en línea")
    print("   + Combina policy gradient con función de valor")
    print("   - Más complejo de implementar")
    
    print("\n3. Hill Climbing:")
    print("   + Muy simple")
    print("   + No requiere gradientes")
    print("   - Puede quedar atrapado en óptimos locales")
    print("   - Ineficiente en espacios grandes")
    
    print("\n4. Algoritmos Genéticos:")
    print("   + Búsqueda global")
    print("   + Paralelizable")
    print("   - Requiere muchas evaluaciones")
    print("   - No usa estructura del problema")
    
    print("\nCuándo usar búsqueda de políticas:")
    print("- Espacios de acción continuos")
    print("- Políticas estocásticas necesarias")
    print("- Cuando la función de valor es difícil de aprender")
    print("- Problemas de alta dimensionalidad")


# Ejecutar ejemplos
if __name__ == "__main__":
    ejemplo_reinforce()
    ejemplo_hill_climbing()
    comparacion_metodos()
    
    print("\n" + "="*70)
    print("\nCaracterísticas de Búsqueda de Políticas:")
    print("- Optimiza directamente la política")
    print("- No requiere función de valor explícita")
    print("- Funciona bien con acciones continuas")
    print("- Puede aprender políticas estocásticas")
    print("\nVentajas:")
    print("- Convergencia a políticas estocásticas")
    print("- Mejor para espacios de acción continuos")
    print("- Puede ser más estable que métodos basados en valor")
    print("\nDesventajas:")
    print("- Puede tener alta varianza")
    print("- Convergencia puede ser lenta")
    print("- Sensible a hiperparámetros")

