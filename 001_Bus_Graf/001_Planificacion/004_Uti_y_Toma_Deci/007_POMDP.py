"""
Algoritmo 30: MDP Parcialmente Observable (POMDP)

Un POMDP extiende los MDPs para situaciones donde el agente no puede
observar directamente el estado completo del sistema.

Componentes adicionales:
- O: Conjunto de observaciones
- Z(o|s',a): Función de observación P(observación | estado, acción)
- b: Creencia (distribución de probabilidad sobre estados)

El agente mantiene una distribución de creencias y toma decisiones
basadas en esta creencia en lugar del estado verdadero.
"""

from typing import Dict, List, Tuple, Optional
import random
import math


class POMDP:
    """Proceso de Decisión de Markov Parcialmente Observable"""
    
    def __init__(self, estados: List[str], acciones: List[str], observaciones: List[str],
                 transiciones: Dict[Tuple[str, str, str], float],
                 observaciones_prob: Dict[Tuple[str, str, str], float],
                 recompensas: Dict[Tuple[str, str, str], float],
                 creencia_inicial: Dict[str, float],
                 gamma: float = 0.9):
        """
        Args:
            estados: Lista de estados (no observables directamente)
            acciones: Lista de acciones
            observaciones: Lista de observaciones posibles
            transiciones: P(s'|s,a)
            observaciones_prob: P(o|s',a) - probabilidad de observar o después de acción a llegando a s'
            recompensas: R(s,a,s')
            creencia_inicial: Distribución inicial sobre estados
            gamma: Factor de descuento
        """
        self.estados = estados
        self.acciones = acciones
        self.observaciones = observaciones
        self.transiciones = transiciones
        self.observaciones_prob = observaciones_prob
        self.recompensas = recompensas
        self.creencia = creencia_inicial.copy()
        self.gamma = gamma
    
    def actualizar_creencia(self, accion: str, observacion: str) -> Dict[str, float]:
        """
        Actualiza la creencia usando el filtro de Bayes.
        
        b'(s') = η * P(o|s',a) * Σ_s P(s'|s,a) * b(s)
        
        Args:
            accion: Acción tomada
            observacion: Observación recibida
        
        Returns:
            Nueva distribución de creencia
        """
        nueva_creencia = {}
        
        for s_nuevo in self.estados:
            # P(o|s',a)
            prob_obs = self.observaciones_prob.get((s_nuevo, accion, observacion), 0.0)
            
            # Σ_s P(s'|s,a) * b(s)
            prob_transicion = 0.0
            for s_viejo in self.estados:
                prob_trans = self.transiciones.get((s_viejo, accion, s_nuevo), 0.0)
                prob_transicion += prob_trans * self.creencia.get(s_viejo, 0.0)
            
            nueva_creencia[s_nuevo] = prob_obs * prob_transicion
        
        # Normalizar
        total = sum(nueva_creencia.values())
        if total > 0:
            nueva_creencia = {s: p/total for s, p in nueva_creencia.items()}
        else:
            # Si no hay probabilidad, distribución uniforme
            nueva_creencia = {s: 1.0/len(self.estados) for s in self.estados}
        
        return nueva_creencia
    
    def recompensa_esperada_creencia(self, creencia: Dict[str, float], accion: str) -> float:
        """Calcula la recompensa esperada dada una creencia y acción"""
        recompensa = 0.0
        
        for s in self.estados:
            for s_sig in self.estados:
                prob_trans = self.transiciones.get((s, accion, s_sig), 0.0)
                r = self.recompensas.get((s, accion, s_sig), 0.0)
                recompensa += creencia.get(s, 0) * prob_trans * r
        
        return recompensa
    
    def mejor_accion_creencia(self, creencia: Dict[str, float]) -> Tuple[str, float]:
        """
        Encuentra la mejor acción dada una creencia (política greedy simple).
        
        Returns:
            Tupla (mejor_accion, valor_esperado)
        """
        mejor_accion = None
        mejor_valor = float('-inf')
        
        for accion in self.acciones:
            valor = self.recompensa_esperada_creencia(creencia, accion)
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion
        
        return mejor_accion, mejor_valor
    
    def ejecutar_paso(self, accion: str, estado_real: str) -> Tuple[str, str, float]:
        """
        Simula un paso en el POMDP.
        
        Args:
            accion: Acción a ejecutar
            estado_real: Estado real (oculto al agente)
        
        Returns:
            Tupla (nuevo_estado, observacion, recompensa)
        """
        # Transición de estado
        transiciones_posibles = [(s, self.transiciones.get((estado_real, accion, s), 0))
                                for s in self.estados]
        estados, probs = zip(*[(s, p) for s, p in transiciones_posibles if p > 0])
        
        if not estados:
            nuevo_estado = estado_real
        else:
            nuevo_estado = random.choices(estados, weights=probs)[0]
        
        # Generar observación
        obs_posibles = [(o, self.observaciones_prob.get((nuevo_estado, accion, o), 0))
                       for o in self.observaciones]
        observaciones, probs_obs = zip(*[(o, p) for o, p in obs_posibles if p > 0])
        
        if not observaciones:
            observacion = random.choice(self.observaciones)
        else:
            observacion = random.choices(observaciones, weights=probs_obs)[0]
        
        # Recompensa
        recompensa = self.recompensas.get((estado_real, accion, nuevo_estado), 0.0)
        
        return nuevo_estado, observacion, recompensa


# Ejemplo: Problema del Tigre
def ejemplo_tigre():
    """
    Problema clásico del Tigre:
    
    Hay dos puertas. Detrás de una hay un tigre (peligroso), detrás de la otra hay un tesoro.
    El agente puede:
    - Escuchar (recibe observación ruidosa)
    - Abrir puerta izquierda
    - Abrir puerta derecha
    
    Estados: {tigre_izquierda, tigre_derecha}
    Acciones: {escuchar, abrir_izq, abrir_der}
    Observaciones: {rugido_izq, rugido_der}
    """
    print("=== Ejemplo: Problema del Tigre ===\n")
    
    estados = ["tigre_izq", "tigre_der"]
    acciones = ["escuchar", "abrir_izq", "abrir_der"]
    observaciones = ["rugido_izq", "rugido_der"]
    
    # Transiciones (el tigre no se mueve al escuchar, se reinicia al abrir)
    transiciones = {}
    for s in estados:
        # Escuchar no cambia el estado
        transiciones[(s, "escuchar", s)] = 1.0
        
        # Abrir puerta reinicia aleatoriamente
        for s_nuevo in estados:
            transiciones[(s, "abrir_izq", s_nuevo)] = 0.5
            transiciones[(s, "abrir_der", s_nuevo)] = 0.5
    
    # Observaciones (85% precisión al escuchar)
    observaciones_prob = {}
    for s in estados:
        for a in acciones:
            if a == "escuchar":
                if s == "tigre_izq":
                    observaciones_prob[(s, a, "rugido_izq")] = 0.85
                    observaciones_prob[(s, a, "rugido_der")] = 0.15
                else:  # tigre_der
                    observaciones_prob[(s, a, "rugido_izq")] = 0.15
                    observaciones_prob[(s, a, "rugido_der")] = 0.85
            else:
                # Al abrir, no hay observación útil
                for o in observaciones:
                    observaciones_prob[(s, a, o)] = 0.5
    
    # Recompensas
    recompensas = {}
    for s in estados:
        for s_sig in estados:
            # Escuchar tiene costo pequeño
            recompensas[(s, "escuchar", s_sig)] = -1.0
            
            # Abrir puerta correcta da recompensa
            if s == "tigre_izq":
                recompensas[(s, "abrir_izq", s_sig)] = -100.0  # Tigre!
                recompensas[(s, "abrir_der", s_sig)] = 10.0    # Tesoro
            else:
                recompensas[(s, "abrir_izq", s_sig)] = 10.0    # Tesoro
                recompensas[(s, "abrir_der", s_sig)] = -100.0  # Tigre!
    
    # Creencia inicial (uniforme)
    creencia_inicial = {"tigre_izq": 0.5, "tigre_der": 0.5}
    
    pomdp = POMDP(estados, acciones, observaciones, transiciones, 
                  observaciones_prob, recompensas, creencia_inicial, gamma=0.95)
    
    # Simular episodio
    print("Simulación de episodio:\n")
    
    # Estado real (desconocido para el agente)
    estado_real = random.choice(estados)
    print(f"Estado real (oculto): {estado_real}\n")
    
    creencia_actual = creencia_inicial.copy()
    
    for paso in range(5):
        print(f"Paso {paso + 1}:")
        print(f"  Creencia: tigre_izq={creencia_actual['tigre_izq']:.3f}, "
              f"tigre_der={creencia_actual['tigre_der']:.3f}")
        
        # Decidir acción basada en creencia
        accion, valor = pomdp.mejor_accion_creencia(creencia_actual)
        print(f"  Acción elegida: {accion} (valor esperado: {valor:.2f})")
        
        # Ejecutar acción
        estado_real, observacion, recompensa = pomdp.ejecutar_paso(accion, estado_real)
        print(f"  Observación: {observacion}")
        print(f"  Recompensa: {recompensa:.1f}")
        
        # Actualizar creencia
        creencia_actual = pomdp.actualizar_creencia(accion, observacion)
        
        # Si abrió una puerta, terminar
        if accion.startswith("abrir"):
            print(f"\n¡Episodio terminado!")
            if recompensa > 0:
                print(f"✓ ¡Encontró el tesoro!")
            else:
                print(f"✗ ¡Encontró el tigre!")
            break
        
        print()


# Ejemplo: Robot con sensores ruidosos
def ejemplo_robot_localizacion():
    """
    Robot que intenta localizarse en un pasillo con sensores ruidosos.
    
    Estados: {pos0, pos1, pos2}
    Acciones: {mover_izq, mover_der, quedarse}
    Observaciones: {sensor_0, sensor_1, sensor_2}
    """
    print("\n\n=== Ejemplo: Localización de Robot ===\n")
    
    estados = ["pos0", "pos1", "pos2"]
    acciones = ["izq", "der", "quedarse"]
    observaciones = ["sensor_0", "sensor_1", "sensor_2"]
    
    # Transiciones (80% éxito, 20% falla)
    transiciones = {}
    for i, s in enumerate(estados):
        # Mover izquierda
        if i > 0:
            transiciones[(s, "izq", estados[i-1])] = 0.8
            transiciones[(s, "izq", s)] = 0.2
        else:
            transiciones[(s, "izq", s)] = 1.0
        
        # Mover derecha
        if i < len(estados) - 1:
            transiciones[(s, "der", estados[i+1])] = 0.8
            transiciones[(s, "der", s)] = 0.2
        else:
            transiciones[(s, "der", s)] = 1.0
        
        # Quedarse
        transiciones[(s, "quedarse", s)] = 1.0
    
    # Observaciones (70% precisión)
    observaciones_prob = {}
    for i, s in enumerate(estados):
        for a in acciones:
            for j, o in enumerate(observaciones):
                if i == j:
                    observaciones_prob[(s, a, o)] = 0.7
                else:
                    observaciones_prob[(s, a, o)] = 0.15
    
    # Recompensas (queremos llegar a pos2)
    recompensas = {}
    for s in estados:
        for a in acciones:
            for s_sig in estados:
                if s_sig == "pos2":
                    recompensas[(s, a, s_sig)] = 10.0
                else:
                    recompensas[(s, a, s_sig)] = -1.0
    
    creencia_inicial = {s: 1.0/len(estados) for s in estados}
    
    pomdp = POMDP(estados, acciones, observaciones, transiciones,
                  observaciones_prob, recompensas, creencia_inicial)
    
    print("El robot intenta localizarse y llegar a pos2\n")
    
    estado_real = "pos0"
    creencia = creencia_inicial.copy()
    
    for paso in range(6):
        print(f"Paso {paso + 1}:")
        print(f"  Creencia: {' '.join([f'{s}={creencia[s]:.2f}' for s in estados])}")
        
        accion, _ = pomdp.mejor_accion_creencia(creencia)
        print(f"  Acción: {accion}")
        
        estado_real, obs, recomp = pomdp.ejecutar_paso(accion, estado_real)
        print(f"  Observación: {obs}, Recompensa: {recomp:.1f}")
        print(f"  Estado real: {estado_real}")
        
        creencia = pomdp.actualizar_creencia(accion, obs)
        print()


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== MDP Parcialmente Observable (POMDP) ===\n")
    
    ejemplo_tigre()
    ejemplo_robot_localizacion()
    
    print("="*70)
    print("\nCaracterísticas de POMDPs:")
    print("- El agente no observa el estado directamente")
    print("- Mantiene distribución de creencias sobre estados")
    print("- Actualiza creencias usando filtro de Bayes")
    print("- Toma decisiones basadas en creencias")
    print("\nComponentes adicionales vs MDP:")
    print("- O: Conjunto de observaciones")
    print("- Z(o|s',a): Función de observación")
    print("- b: Distribución de creencias")
    print("\nAplicaciones:")
    print("- Robótica con sensores ruidosos")
    print("- Juegos de información imperfecta")
    print("- Sistemas de diálogo")

