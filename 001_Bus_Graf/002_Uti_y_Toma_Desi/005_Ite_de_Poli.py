"""
Algoritmo 28: Iteración de Políticas (Policy Iteration)

La iteración de políticas es otro algoritmo para resolver MDPs.
Alterna entre dos pasos:
1. Evaluación de política: Calcular V^π(s) para la política actual
2. Mejora de política: Actualizar π para ser greedy respecto a V^π

A menudo converge en menos iteraciones que iteración de valores.
"""

from typing import Dict, List, Tuple
import random


class MDP:
    """Proceso de Decisión de Markov"""
    
    def __init__(self, estados, acciones, transiciones, recompensas, gamma=0.9):
        self.estados = estados
        self.acciones = acciones
        self.transiciones = transiciones
        self.recompensas = recompensas
        self.gamma = gamma
    
    def obtener_transiciones(self, estado, accion):
        transiciones = []
        for s_siguiente in self.estados:
            prob = self.transiciones.get((estado, accion, s_siguiente), 0.0)
            if prob > 0:
                transiciones.append((s_siguiente, prob))
        return transiciones
    
    def obtener_recompensa(self, estado, accion, estado_siguiente):
        return self.recompensas.get((estado, accion, estado_siguiente), 0.0)


def evaluacion_politica(mdp: MDP, politica: Dict[str, str], 
                        epsilon: float = 0.01, max_iter: int = 1000) -> Dict[str, float]:
    """
    Evalúa una política calculando V^π(s).
    
    Args:
        mdp: El MDP
        politica: Política π(s) a evaluar
        epsilon: Criterio de convergencia
        max_iter: Máximo número de iteraciones
    
    Returns:
        Función de valor V^π
    """
    V = {s: 0.0 for s in mdp.estados}
    
    for _ in range(max_iter):
        delta = 0.0
        V_nuevo = {}
        
        for s in mdp.estados:
            # Acción dictada por la política
            a = politica[s]
            
            # Calcular valor esperado
            valor = 0.0
            for s_siguiente, prob in mdp.obtener_transiciones(s, a):
                recompensa = mdp.obtener_recompensa(s, a, s_siguiente)
                valor += prob * (recompensa + mdp.gamma * V[s_siguiente])
            
            V_nuevo[s] = valor
            delta = max(delta, abs(V_nuevo[s] - V[s]))
        
        V = V_nuevo
        
        if delta < epsilon:
            break
    
    return V


def mejora_politica(mdp: MDP, V: Dict[str, float]) -> Dict[str, str]:
    """
    Mejora la política siendo greedy respecto a V.
    
    Args:
        mdp: El MDP
        V: Función de valor
    
    Returns:
        Nueva política mejorada
    """
    politica_nueva = {}
    
    for s in mdp.estados:
        mejor_accion = None
        mejor_valor = float('-inf')
        
        for a in mdp.acciones:
            valor_accion = 0.0
            for s_siguiente, prob in mdp.obtener_transiciones(s, a):
                recompensa = mdp.obtener_recompensa(s, a, s_siguiente)
                valor_accion += prob * (recompensa + mdp.gamma * V[s_siguiente])
            
            if valor_accion > mejor_valor:
                mejor_valor = valor_accion
                mejor_accion = a
        
        politica_nueva[s] = mejor_accion
    
    return politica_nueva


def iteracion_politicas(mdp: MDP, max_iteraciones: int = 100) -> Tuple[Dict[str, str], Dict[str, float], int]:
    """
    Algoritmo de iteración de políticas.
    
    Args:
        mdp: El MDP a resolver
        max_iteraciones: Máximo número de iteraciones
    
    Returns:
        Tupla (política_óptima, V_óptima, iteraciones)
    """
    # Inicializar política aleatoriamente
    politica = {s: random.choice(mdp.acciones) for s in mdp.estados}
    
    for iteracion in range(max_iteraciones):
        # Paso 1: Evaluación de política
        V = evaluacion_politica(mdp, politica)
        
        # Paso 2: Mejora de política
        politica_nueva = mejora_politica(mdp, V)
        
        # Verificar convergencia
        if politica_nueva == politica:
            return politica, V, iteracion + 1
        
        politica = politica_nueva
    
    # Si no convergió, retornar última política
    V = evaluacion_politica(mdp, politica)
    return politica, V, max_iteraciones


# Ejemplo: Mundo de cuadrícula
def ejemplo_mundo_cuadricula():
    """Mismo ejemplo que en iteración de valores"""
    print("=== Ejemplo: Mundo de Cuadrícula ===\n")
    
    estados = [
        "(0,0)", "(1,0)", "(2,0)",
        "(0,1)", "(1,1)", "(2,1)",
        "(0,2)", "(1,2)", "(2,2)"
    ]
    
    terminales = {"(2,0)", "(0,2)"}
    obstaculo = "(1,1)"
    acciones = ["arriba", "abajo", "izquierda", "derecha"]
    
    def siguiente_estado(s, a):
        if s in terminales or s == obstaculo:
            return s
        
        x, y = eval(s)
        
        if a == "arriba":
            y = max(0, y - 1)
        elif a == "abajo":
            y = min(2, y + 1)
        elif a == "izquierda":
            x = max(0, x - 1)
        elif a == "derecha":
            x = min(2, x + 1)
        
        nuevo_s = f"({x},{y})"
        if nuevo_s == obstaculo:
            return s
        return nuevo_s
    
    transiciones = {}
    for s in estados:
        for a in acciones:
            s_sig = siguiente_estado(s, a)
            transiciones[(s, a, s_sig)] = 1.0
    
    recompensas = {}
    for s in estados:
        for a in acciones:
            s_sig = siguiente_estado(s, a)
            if s_sig == "(2,0)":
                recompensas[(s, a, s_sig)] = 10.0
            elif s_sig == "(0,2)":
                recompensas[(s, a, s_sig)] = -10.0
            else:
                recompensas[(s, a, s_sig)] = -0.1
    
    mdp = MDP(estados, acciones, transiciones, recompensas, gamma=0.9)
    
    # Resolver con iteración de políticas
    print("Resolviendo con Iteración de Políticas...")
    politica, V, iteraciones = iteracion_politicas(mdp)
    
    print(f"Convergencia en {iteraciones} iteraciones\n")
    
    # Mostrar función de valor
    print("Función de Valor V*(s):")
    for y in range(3):
        fila = []
        for x in range(3):
            s = f"({x},{y})"
            if s == obstaculo:
                fila.append("  X  ")
            else:
                fila.append(f"{V[s]:5.2f}")
        print("  ".join(fila))
    
    # Mostrar política
    print("\nPolítica Óptima π*(s):")
    simbolos = {
        "arriba": "↑",
        "abajo": "↓",
        "izquierda": "←",
        "derecha": "→"
    }
    
    for y in range(3):
        fila = []
        for x in range(3):
            s = f"({x},{y})"
            if s in terminales:
                fila.append(" * ")
            elif s == obstaculo:
                fila.append(" X ")
            else:
                fila.append(f" {simbolos.get(politica[s], '?')} ")
        print("  ".join(fila))


# Comparación con iteración de valores
def comparacion_algoritmos():
    """Compara iteración de valores vs iteración de políticas"""
    print("\n\n=== Comparación: Value Iteration vs Policy Iteration ===\n")
    
    # MDP simple para comparación
    estados = ["s1", "s2", "s3"]
    acciones = ["a1", "a2"]
    
    transiciones = {
        ("s1", "a1", "s2"): 0.8,
        ("s1", "a1", "s3"): 0.2,
        ("s1", "a2", "s1"): 1.0,
        ("s2", "a1", "s3"): 1.0,
        ("s2", "a2", "s1"): 1.0,
        ("s3", "a1", "s3"): 1.0,
        ("s3", "a2", "s3"): 1.0,
    }
    
    recompensas = {
        ("s1", "a1", "s2"): 10.0,
        ("s1", "a1", "s3"): 5.0,
        ("s1", "a2", "s1"): 0.0,
        ("s2", "a1", "s3"): 5.0,
        ("s2", "a2", "s1"): -1.0,
        ("s3", "a1", "s3"): 0.0,
        ("s3", "a2", "s3"): 0.0,
    }
    
    mdp = MDP(estados, acciones, transiciones, recompensas, gamma=0.9)
    
    # Iteración de políticas
    politica_pi, V_pi, iter_pi = iteracion_politicas(mdp)
    
    print(f"Iteración de Políticas:")
    print(f"  Iteraciones: {iter_pi}")
    print(f"  Política: {politica_pi}")
    print(f"  Valores: {V_pi}")
    
    # Para comparar con VI, necesitamos importarlo
    # Por simplicidad, solo mostramos PI aquí
    
    print("\n" + "="*70)
    print("\nCaracterísticas comparativas:")
    print("\nIteración de Políticas:")
    print("  + Generalmente menos iteraciones")
    print("  + Cada iteración es más costosa (evaluación completa)")
    print("  + Siempre produce política válida")
    print("\nIteración de Valores:")
    print("  + Cada iteración es más rápida")
    print("  + Generalmente más iteraciones totales")
    print("  + Política se extrae al final")


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Iteración de Políticas (Policy Iteration) ===\n")
    
    ejemplo_mundo_cuadricula()
    comparacion_algoritmos()
    
    print("\n" + "="*70)
    print("\nAlgoritmo de Iteración de Políticas:")
    print("1. Inicializar política π arbitrariamente")
    print("2. Repetir:")
    print("   a) Evaluación: Calcular V^π")
    print("   b) Mejora: π' = greedy(V^π)")
    print("   c) Si π' = π, terminar")
    print("\nComplejidad:")
    print("- Evaluación: O(|S|²|A|) por iteración")
    print("- Mejora: O(|S||A|)")
    print("- Generalmente converge en pocas iteraciones")

