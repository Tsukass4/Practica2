"""
Algoritmo 35: Q-Learning

Q-Learning es un algoritmo de aprendizaje por refuerzo libre de modelo (model-free)
y off-policy. Aprende la función de valor acción-estado óptima Q*(s,a) directamente
de la experiencia.

Actualización:
Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]

Características:
- Off-policy: Aprende política óptima independientemente de la política seguida
- Libre de modelo: No requiere conocer P(s'|s,a) ni R(s,a,s')
- Converge a Q* bajo ciertas condiciones
"""

from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict
import numpy as np


class QLearning:
    """Algoritmo Q-Learning"""
    
    def __init__(self, acciones: List[str], alpha: float = 0.1, 
                 gamma: float = 0.9, epsilon_inicial: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995):
        """
        Args:
            acciones: Lista de acciones posibles
            alpha: Tasa de aprendizaje
            gamma: Factor de descuento
            epsilon_inicial: Epsilon inicial para exploración
            epsilon_min: Epsilon mínimo
            epsilon_decay: Factor de decaimiento de epsilon
        """
        self.acciones = acciones
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_inicial
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Tabla Q
        self.Q = defaultdict(float)
        
        # Estadísticas
        self.episodios_entrenados = 0
        self.pasos_totales = 0
    
    def elegir_accion(self, estado, entrenar: bool = True) -> str:
        """
        Elige acción usando epsilon-greedy.
        
        Args:
            estado: Estado actual
            entrenar: Si es False, usa política greedy pura (epsilon=0)
        """
        epsilon_actual = self.epsilon if entrenar else 0.0
        
        if random.random() < epsilon_actual:
            # Exploración: acción aleatoria
            return random.choice(self.acciones)
        else:
            # Explotación: mejor acción conocida
            valores_q = [(a, self.Q[(estado, a)]) for a in self.acciones]
            max_q = max(valores_q, key=lambda x: x[1])[1]
            
            # Si hay empates, elegir aleatoriamente entre ellos
            mejores_acciones = [a for a, q in valores_q if q == max_q]
            return random.choice(mejores_acciones)
    
    def actualizar(self, estado, accion, recompensa, estado_siguiente, terminal: bool):
        """
        Actualización Q-Learning:
        Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]
        """
        if terminal:
            objetivo = recompensa
        else:
            # Max sobre acciones en el siguiente estado
            max_q_siguiente = max(self.Q[(estado_siguiente, a)] for a in self.acciones)
            objetivo = recompensa + self.gamma * max_q_siguiente
        
        # Error TD
        error_td = objetivo - self.Q[(estado, accion)]
        
        # Actualizar Q
        self.Q[(estado, accion)] += self.alpha * error_td
        
        self.pasos_totales += 1
        
        return error_td
    
    def entrenar_episodio(self, entorno, max_pasos: int = 100) -> Tuple[float, int]:
        """
        Entrena un episodio completo.
        
        Returns:
            Tupla (recompensa_total, pasos)
        """
        estado = entorno.reiniciar()
        recompensa_total = 0.0
        pasos = 0
        
        for paso in range(max_pasos):
            # Elegir y ejecutar acción
            accion = self.elegir_accion(estado)
            estado_siguiente, recompensa, terminal = entorno.ejecutar_accion(accion)
            
            # Actualizar Q
            self.actualizar(estado, accion, recompensa, estado_siguiente, terminal)
            
            recompensa_total += recompensa
            pasos += 1
            estado = estado_siguiente
            
            if terminal:
                break
        
        # Decaer epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodios_entrenados += 1
        
        return recompensa_total, pasos
    
    def obtener_politica(self) -> Dict:
        """Extrae la política greedy de la tabla Q"""
        politica = {}
        estados = set(s for s, a in self.Q.keys())
        
        for estado in estados:
            valores_q = [(a, self.Q[(estado, a)]) for a in self.acciones]
            mejor_accion = max(valores_q, key=lambda x: x[1])[0]
            politica[estado] = mejor_accion
        
        return politica


class EntornoGridWorld:
    """Mundo de cuadrícula para Q-Learning"""
    
    def __init__(self):
        self.filas = 3
        self.columnas = 3
        self.estado_actual = (0, 0)
        self.terminales = {(2, 2): 10.0, (2, 0): -10.0}
        self.obstaculos = {(1, 1)}
        self.acciones = ['arriba', 'abajo', 'izquierda', 'derecha']
    
    def reiniciar(self) -> Tuple[int, int]:
        self.estado_actual = (0, 0)
        return self.estado_actual
    
    def es_terminal(self, estado: Tuple[int, int]) -> bool:
        return estado in self.terminales
    
    def ejecutar_accion(self, accion: str) -> Tuple[Tuple[int, int], float, bool]:
        if self.es_terminal(self.estado_actual):
            return self.estado_actual, 0.0, True
        
        fila, col = self.estado_actual
        
        # 80% de éxito
        if random.random() < 0.8:
            accion_real = accion
        else:
            accion_real = random.choice(self.acciones)
        
        if accion_real == 'arriba':
            nuevo_estado = (max(0, fila - 1), col)
        elif accion_real == 'abajo':
            nuevo_estado = (min(self.filas - 1, fila + 1), col)
        elif accion_real == 'izquierda':
            nuevo_estado = (fila, max(0, col - 1))
        else:
            nuevo_estado = (fila, min(self.columnas - 1, col + 1))
        
        if nuevo_estado in self.obstaculos:
            nuevo_estado = self.estado_actual
        
        if nuevo_estado in self.terminales:
            recompensa = self.terminales[nuevo_estado]
        else:
            recompensa = -0.1
        
        self.estado_actual = nuevo_estado
        return nuevo_estado, recompensa, self.es_terminal(nuevo_estado)


# Ejemplo de uso
def ejemplo_q_learning():
    """Entrena un agente Q-Learning en GridWorld"""
    print("=== Q-Learning ===\n")
    
    # Crear entorno y agente
    entorno = EntornoGridWorld()
    agente = QLearning(
        acciones=entorno.acciones,
        alpha=0.1,
        gamma=0.9,
        epsilon_inicial=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Entrenar
    print("Entrenando agente Q-Learning...")
    num_episodios = 500
    recompensas_por_episodio = []
    
    for episodio in range(num_episodios):
        recompensa_total, pasos = agente.entrenar_episodio(entorno)
        recompensas_por_episodio.append(recompensa_total)
        
        if (episodio + 1) % 100 == 0:
            promedio_reciente = np.mean(recompensas_por_episodio[-100:])
            print(f"Episodio {episodio + 1}: Recompensa promedio (últimos 100) = {promedio_reciente:.2f}, "
                  f"Epsilon = {agente.epsilon:.3f}")
    
    # Mostrar política aprendida
    print("\nPolítica aprendida:")
    politica = agente.obtener_politica()
    simbolos = {'arriba': '↑', 'abajo': '↓', 'izquierda': '←', 'derecha': '→'}
    
    for fila in range(3):
        acciones_fila = []
        for col in range(3):
            estado = (fila, col)
            if estado in entorno.obstaculos:
                acciones_fila.append(" X ")
            elif estado in entorno.terminales:
                acciones_fila.append(" * ")
            else:
                accion = politica.get(estado, '?')
                acciones_fila.append(f" {simbolos.get(accion, '?')} ")
        print("  ".join(acciones_fila))
    
    # Mostrar valores Q
    print("\nValores Q para estado (0,0):")
    for accion in entorno.acciones:
        q_valor = agente.Q[((0, 0), accion)]
        print(f"  Q((0,0), {accion:10s}) = {q_valor:6.3f}")
    
    # Evaluar política aprendida
    print("\nEvaluando política aprendida (10 episodios de prueba)...")
    recompensas_prueba = []
    
    for _ in range(10):
        entorno.reiniciar()
        recompensa_episodio = 0.0
        
        for paso in range(50):
            estado = entorno.estado_actual
            accion = agente.elegir_accion(estado, entrenar=False)  # Sin exploración
            estado_sig, recompensa, terminal = entorno.ejecutar_accion(accion)
            recompensa_episodio += recompensa
            
            if terminal:
                break
        
        recompensas_prueba.append(recompensa_episodio)
    
    print(f"Recompensa promedio en prueba: {np.mean(recompensas_prueba):.2f} ± {np.std(recompensas_prueba):.2f}")


# Comparación con SARSA
def comparacion_q_learning_sarsa():
    """Compara Q-Learning (off-policy) con SARSA (on-policy)"""
    print("\n\n=== Comparación: Q-Learning vs SARSA ===\n")
    
    print("Diferencias clave:")
    print("\nQ-Learning (Off-policy):")
    print("  - Actualización: Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]")
    print("  - Aprende política óptima independiente de la política seguida")
    print("  - Puede ser más agresivo en la exploración")
    print("  - Converge a Q* incluso con política exploratoria")
    print("\nSARSA (On-policy):")
    print("  - Actualización: Q(s,a) ← Q(s,a) + α[R + γ Q(s',a') - Q(s,a)]")
    print("  - Aprende el valor de la política que está siguiendo")
    print("  - Más conservador, considera la exploración")
    print("  - Mejor en entornos con riesgo durante exploración")
    
    print("\nCuándo usar cada uno:")
    print("  Q-Learning: Cuando se puede explorar libremente sin consecuencias graves")
    print("  SARSA: Cuando la exploración es arriesgada (ej: robots físicos)")


# Ejecutar ejemplos
if __name__ == "__main__":
    ejemplo_q_learning()
    comparacion_q_learning_sarsa()
    
    print("\n" + "="*70)
    print("\nCaracterísticas de Q-Learning:")
    print("- Algoritmo off-policy: Aprende Q* independiente de π")
    print("- Libre de modelo: No requiere conocer transiciones")
    print("- Converge a política óptima bajo condiciones adecuadas")
    print("- Requiere exploración suficiente")
    print("\nVentajas:")
    print("- Simple de implementar")
    print("- No requiere modelo del entorno")
    print("- Aprende política óptima directamente")
    print("\nDesventajas:")
    print("- Puede ser lento en espacios grandes")
    print("- Requiere discretización para estados continuos")
    print("- Tabla Q puede ser muy grande")

