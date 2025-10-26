"""
Algoritmo 34: Aprendizaje por Refuerzo Activo

En el aprendizaje activo, el agente aprende mientras explora y mejora
su política simultáneamente. A diferencia del pasivo, el agente elige
sus propias acciones para maximizar la recompensa.

Métodos:
- ADP (Adaptive Dynamic Programming): Aprende modelo y resuelve MDP
- Q-Learning: Aprende función Q sin modelo
- SARSA: On-policy TD control
"""

from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict
import math


class EntornoGridWorld:
    """Mundo de cuadrícula para aprendizaje activo"""
    
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
        """Ejecuta acción con 80% de éxito"""
        if self.es_terminal(self.estado_actual):
            return self.estado_actual, 0.0, True
        
        fila, col = self.estado_actual
        
        # 80% de éxito, 20% aleatorio
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


class AprendizajeActivoADP:
    """
    Adaptive Dynamic Programming (ADP):
    Aprende el modelo del entorno y resuelve el MDP.
    """
    
    def __init__(self, acciones: List[str], gamma: float = 0.9):
        self.acciones = acciones
        self.gamma = gamma
        
        # Modelo aprendido
        self.transiciones = defaultdict(lambda: defaultdict(int))  # Conteos
        self.recompensas = defaultdict(lambda: defaultdict(float))  # Suma de recompensas
        
        # Función de valor y política
        self.V = defaultdict(float)
        self.politica = {}
    
    def actualizar_modelo(self, estado, accion, recompensa, estado_siguiente):
        """Actualiza el modelo del entorno"""
        self.transiciones[(estado, accion)][estado_siguiente] += 1
        self.recompensas[(estado, accion)][estado_siguiente] += recompensa
    
    def obtener_prob_transicion(self, estado, accion, estado_siguiente) -> float:
        """Estima P(s'|s,a) del modelo aprendido"""
        total = sum(self.transiciones[(estado, accion)].values())
        if total == 0:
            return 0.0
        return self.transiciones[(estado, accion)][estado_siguiente] / total
    
    def obtener_recompensa_esperada(self, estado, accion, estado_siguiente) -> float:
        """Estima R(s,a,s') del modelo aprendido"""
        count = self.transiciones[(estado, accion)][estado_siguiente]
        if count == 0:
            return 0.0
        return self.recompensas[(estado, accion)][estado_siguiente] / count
    
    def resolver_mdp(self, estados_conocidos: set, iteraciones: int = 10):
        """Resuelve el MDP aprendido usando iteración de valores"""
        for _ in range(iteraciones):
            V_nuevo = defaultdict(float)
            
            for estado in estados_conocidos:
                valores_acciones = []
                
                for accion in self.acciones:
                    valor = 0.0
                    for estado_sig in self.transiciones[(estado, accion)]:
                        prob = self.obtener_prob_transicion(estado, accion, estado_sig)
                        recompensa = self.obtener_recompensa_esperada(estado, accion, estado_sig)
                        valor += prob * (recompensa + self.gamma * self.V[estado_sig])
                    valores_acciones.append((accion, valor))
                
                if valores_acciones:
                    mejor_accion, mejor_valor = max(valores_acciones, key=lambda x: x[1])
                    V_nuevo[estado] = mejor_valor
                    self.politica[estado] = mejor_accion
            
            self.V = V_nuevo
    
    def elegir_accion(self, estado, epsilon: float = 0.1) -> str:
        """Elige acción usando epsilon-greedy"""
        if random.random() < epsilon or estado not in self.politica:
            return random.choice(self.acciones)
        return self.politica[estado]


class SARSA:
    """
    SARSA: On-policy TD control
    Actualización: Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
    """
    
    def __init__(self, acciones: List[str], alpha: float = 0.1, gamma: float = 0.9):
        self.acciones = acciones
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(float)
    
    def elegir_accion(self, estado, epsilon: float = 0.1) -> str:
        """Epsilon-greedy"""
        if random.random() < epsilon:
            return random.choice(self.acciones)
        
        # Greedy
        valores_q = [(a, self.Q[(estado, a)]) for a in self.acciones]
        return max(valores_q, key=lambda x: x[1])[0]
    
    def actualizar(self, estado, accion, recompensa, estado_sig, accion_sig, terminal):
        """Actualización SARSA"""
        if terminal:
            objetivo = recompensa
        else:
            objetivo = recompensa + self.gamma * self.Q[(estado_sig, accion_sig)]
        
        error_td = objetivo - self.Q[(estado, accion)]
        self.Q[(estado, accion)] += self.alpha * error_td


# Ejemplo de uso
def ejemplo_aprendizaje_activo():
    """Compara ADP y SARSA"""
    print("=== Aprendizaje por Refuerzo Activo ===\n")
    
    # Entrenar con ADP
    print("--- Adaptive Dynamic Programming (ADP) ---")
    entorno = EntornoGridWorld()
    agente_adp = AprendizajeActivoADP(entorno.acciones, gamma=0.9)
    
    estados_visitados = set()
    
    for episodio in range(200):
        entorno.reiniciar()
        epsilon = 0.3 * (1 - episodio / 200)  # Decaimiento de epsilon
        
        for paso in range(50):
            estado = entorno.estado_actual
            estados_visitados.add(estado)
            
            accion = agente_adp.elegir_accion(estado, epsilon)
            estado_sig, recompensa, terminal = entorno.ejecutar_accion(accion)
            
            agente_adp.actualizar_modelo(estado, accion, recompensa, estado_sig)
            
            if terminal:
                break
        
        # Resolver MDP cada 10 episodios
        if episodio % 10 == 0:
            agente_adp.resolver_mdp(estados_visitados)
    
    print("Política aprendida (ADP):")
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
                accion = agente_adp.politica.get(estado, '?')
                acciones_fila.append(f" {simbolos.get(accion, '?')} ")
        print("  ".join(acciones_fila))
    
    # Entrenar con SARSA
    print("\n--- SARSA ---")
    entorno2 = EntornoGridWorld()
    agente_sarsa = SARSA(entorno2.acciones, alpha=0.1, gamma=0.9)
    
    for episodio in range(200):
        entorno2.reiniciar()
        epsilon = 0.3 * (1 - episodio / 200)
        
        estado = entorno2.estado_actual
        accion = agente_sarsa.elegir_accion(estado, epsilon)
        
        for paso in range(50):
            estado_sig, recompensa, terminal = entorno2.ejecutar_accion(accion)
            
            if terminal:
                agente_sarsa.actualizar(estado, accion, recompensa, estado_sig, None, True)
                break
            
            accion_sig = agente_sarsa.elegir_accion(estado_sig, epsilon)
            agente_sarsa.actualizar(estado, accion, recompensa, estado_sig, accion_sig, False)
            
            estado = estado_sig
            accion = accion_sig
    
    print("Política aprendida (SARSA):")
    for fila in range(3):
        acciones_fila = []
        for col in range(3):
            estado = (fila, col)
            if estado in entorno2.obstaculos:
                acciones_fila.append(" X ")
            elif estado in entorno2.terminales:
                acciones_fila.append(" * ")
            else:
                valores_q = [(a, agente_sarsa.Q[(estado, a)]) for a in entorno2.acciones]
                mejor_accion = max(valores_q, key=lambda x: x[1])[0]
                acciones_fila.append(f" {simbolos.get(mejor_accion, '?')} ")
        print("  ".join(acciones_fila))


# Ejecutar ejemplo
if __name__ == "__main__":
    ejemplo_aprendizaje_activo()
    
    print("\n" + "="*70)
    print("\nCaracterísticas del Aprendizaje Activo:")
    print("- El agente elige sus propias acciones")
    print("- Aprende y mejora la política simultáneamente")
    print("- Balance entre exploración y explotación")
    print("\nMétodos principales:")
    print("- ADP: Aprende modelo del entorno y lo resuelve")
    print("- SARSA: On-policy, aprende Q(s,a) de la política seguida")
    print("- Q-Learning: Off-policy, aprende Q óptima")
    print("\nDiferencias:")
    print("- ADP: Basado en modelo (model-based)")
    print("- SARSA/Q-Learning: Libre de modelo (model-free)")
    print("- SARSA: On-policy (aprende de lo que hace)")
    print("- Q-Learning: Off-policy (aprende política óptima)")

