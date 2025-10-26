"""
Algoritmo 29: Proceso de Decisión de Markov (MDP - Markov Decision Process)

Un MDP es un modelo matemático para toma de decisiones secuenciales
en entornos estocásticos. Combina teoría de probabilidad con optimización.

Componentes:
- S: Conjunto de estados
- A: Conjunto de acciones
- P(s'|s,a): Función de transición
- R(s,a,s'): Función de recompensa
- γ: Factor de descuento

Este archivo proporciona una implementación completa de MDPs con
múltiples métodos de solución y ejemplos.
"""

from typing import Dict, List, Tuple, Optional, Set
import random
import math


class MDPCompleto:
    """Implementación completa de un MDP"""
    
    def __init__(self, estados: List[str], acciones: List[str],
                 transiciones: Dict[Tuple[str, str, str], float],
                 recompensas: Dict[Tuple[str, str, str], float],
                 gamma: float = 0.9,
                 estado_inicial: Optional[str] = None):
        self.estados = estados
        self.acciones = acciones
        self.transiciones = transiciones
        self.recompensas = recompensas
        self.gamma = gamma
        self.estado_inicial = estado_inicial or estados[0]
        self.estado_actual = self.estado_inicial
    
    def obtener_transiciones(self, estado: str, accion: str) -> List[Tuple[str, float]]:
        """Retorna lista de (estado_siguiente, probabilidad)"""
        trans = []
        for s_sig in self.estados:
            prob = self.transiciones.get((estado, accion, s_sig), 0.0)
            if prob > 0:
                trans.append((s_sig, prob))
        return trans
    
    def obtener_recompensa(self, estado: str, accion: str, estado_sig: str) -> float:
        """Retorna R(s,a,s')"""
        return self.recompensas.get((estado, accion, estado_sig), 0.0)
    
    def obtener_acciones_validas(self, estado: str) -> List[str]:
        """Retorna acciones válidas en un estado"""
        acciones_validas = []
        for a in self.acciones:
            if any(self.transiciones.get((estado, a, s), 0) > 0 for s in self.estados):
                acciones_validas.append(a)
        return acciones_validas if acciones_validas else self.acciones
    
    def simular_transicion(self, estado: str, accion: str) -> Tuple[str, float]:
        """Simula una transición estocástica"""
        transiciones = self.obtener_transiciones(estado, accion)
        
        if not transiciones:
            return estado, 0.0
        
        # Muestrear siguiente estado
        estados_sig, probs = zip(*transiciones)
        estado_siguiente = random.choices(estados_sig, weights=probs)[0]
        recompensa = self.obtener_recompensa(estado, accion, estado_siguiente)
        
        return estado_siguiente, recompensa
    
    def ejecutar_episodio(self, politica: Dict[str, str], 
                         max_pasos: int = 100) -> Tuple[List[str], List[str], List[float], float]:
        """
        Ejecuta un episodio completo siguiendo una política.
        
        Returns:
            Tupla (estados, acciones, recompensas, retorno_total)
        """
        estados_visitados = [self.estado_inicial]
        acciones_tomadas = []
        recompensas_obtenidas = []
        
        estado_actual = self.estado_inicial
        retorno_total = 0.0
        factor_descuento = 1.0
        
        for paso in range(max_pasos):
            # Seleccionar acción según política
            accion = politica.get(estado_actual, random.choice(self.acciones))
            
            # Simular transición
            estado_siguiente, recompensa = self.simular_transicion(estado_actual, accion)
            
            # Registrar
            acciones_tomadas.append(accion)
            recompensas_obtenidas.append(recompensa)
            estados_visitados.append(estado_siguiente)
            
            # Actualizar retorno
            retorno_total += factor_descuento * recompensa
            factor_descuento *= self.gamma
            
            estado_actual = estado_siguiente
        
        return estados_visitados, acciones_tomadas, recompensas_obtenidas, retorno_total
    
    def evaluar_politica_montecarlo(self, politica: Dict[str, str], 
                                    num_episodios: int = 1000) -> Dict[str, float]:
        """Evalúa una política usando Monte Carlo"""
        retornos = {s: [] for s in self.estados}
        
        for _ in range(num_episodios):
            estados, _, recompensas, _ = self.ejecutar_episodio(politica)
            
            # Calcular retornos para cada visita
            G = 0
            for t in range(len(recompensas) - 1, -1, -1):
                G = recompensas[t] + self.gamma * G
                retornos[estados[t]].append(G)
        
        # Promediar retornos
        V = {}
        for s in self.estados:
            if retornos[s]:
                V[s] = sum(retornos[s]) / len(retornos[s])
            else:
                V[s] = 0.0
        
        return V


# Ejemplo 1: Problema del estudiante
def ejemplo_estudiante():
    """
    Problema del estudiante:
    - Estados: {descansado, cansado}
    - Acciones: {estudiar, dormir, fiesta}
    - Objetivo: Maximizar aprendizaje a largo plazo
    """
    print("=== Ejemplo: Problema del Estudiante ===\n")
    
    estados = ["descansado", "cansado"]
    acciones = ["estudiar", "dormir", "fiesta"]
    
    # Transiciones
    transiciones = {
        # Descansado
        ("descansado", "estudiar", "descansado"): 0.6,
        ("descansado", "estudiar", "cansado"): 0.4,
        ("descansado", "dormir", "descansado"): 1.0,
        ("descansado", "fiesta", "cansado"): 1.0,
        
        # Cansado
        ("cansado", "estudiar", "cansado"): 1.0,
        ("cansado", "dormir", "descansado"): 1.0,
        ("cansado", "fiesta", "cansado"): 1.0,
    }
    
    # Recompensas (aprendizaje)
    recompensas = {
        ("descansado", "estudiar", "descansado"): 10.0,
        ("descansado", "estudiar", "cansado"): 10.0,
        ("descansado", "dormir", "descansado"): 0.0,
        ("descansado", "fiesta", "cansado"): 2.0,
        ("cansado", "estudiar", "cansado"): 2.0,  # Poco efectivo
        ("cansado", "dormir", "descansado"): 0.0,
        ("cansado", "fiesta", "cansado"): 1.0,
    }
    
    mdp = MDPCompleto(estados, acciones, transiciones, recompensas, gamma=0.9)
    
    # Resolver con iteración de valores (implementación simple)
    V = {s: 0.0 for s in estados}
    
    for _ in range(100):
        V_nuevo = {}
        for s in estados:
            valores = []
            for a in acciones:
                valor = sum(
                    prob * (mdp.obtener_recompensa(s, a, s_sig) + mdp.gamma * V[s_sig])
                    for s_sig, prob in mdp.obtener_transiciones(s, a)
                )
                valores.append(valor)
            V_nuevo[s] = max(valores) if valores else 0
        V = V_nuevo
    
    # Extraer política
    politica = {}
    for s in estados:
        mejor_accion = None
        mejor_valor = float('-inf')
        for a in acciones:
            valor = sum(
                prob * (mdp.obtener_recompensa(s, a, s_sig) + mdp.gamma * V[s_sig])
                for s_sig, prob in mdp.obtener_transiciones(s, a)
            )
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = a
        politica[s] = mejor_accion
    
    print("Función de Valor:")
    for s in estados:
        print(f"  V*({s}) = {V[s]:.2f}")
    
    print("\nPolítica Óptima:")
    for s in estados:
        print(f"  π*({s}) = {politica[s]}")
    
    # Simular episodio
    print("\nSimulación de episodio (10 pasos):")
    estados_ep, acciones_ep, recomp_ep, retorno = mdp.ejecutar_episodio(politica, max_pasos=10)
    
    for t in range(len(acciones_ep)):
        print(f"  t={t}: {estados_ep[t]} → {acciones_ep[t]} → {estados_ep[t+1]} (R={recomp_ep[t]:.1f})")
    print(f"\nRetorno total: {retorno:.2f}")


# Ejemplo 2: Inventario
def ejemplo_inventario():
    """
    Problema de gestión de inventario:
    - Estados: Nivel de inventario {0, 1, 2, 3}
    - Acciones: Cantidad a pedir {0, 1, 2, 3}
    - Demanda estocástica
    """
    print("\n\n=== Ejemplo: Gestión de Inventario ===\n")
    
    capacidad = 3
    estados = [str(i) for i in range(capacidad + 1)]
    acciones = [str(i) for i in range(capacidad + 1)]
    
    # Parámetros
    costo_pedido = 2.0
    costo_almacenamiento = 0.5
    precio_venta = 5.0
    prob_demanda = {0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2}
    
    transiciones = {}
    recompensas = {}
    
    for inventario in range(capacidad + 1):
        for pedido in range(capacidad + 1):
            # Inventario después de pedir
            inv_despues_pedido = min(inventario + pedido, capacidad)
            
            for demanda in range(capacidad + 1):
                # Inventario final
                ventas = min(inv_despues_pedido, demanda)
                inv_final = inv_despues_pedido - ventas
                
                # Transición
                s = str(inventario)
                a = str(pedido)
                s_sig = str(inv_final)
                prob = prob_demanda.get(demanda, 0)
                
                transiciones[(s, a, s_sig)] = transiciones.get((s, a, s_sig), 0) + prob
                
                # Recompensa
                ingreso = ventas * precio_venta
                costo = pedido * costo_pedido + inv_final * costo_almacenamiento
                recompensa = ingreso - costo
                
                # Promediar recompensa ponderada por probabilidad
                recompensas[(s, a, s_sig)] = recompensas.get((s, a, s_sig), 0) + prob * recompensa / prob
    
    mdp = MDPCompleto(estados, acciones, transiciones, recompensas, gamma=0.95)
    
    # Resolver
    V = {s: 0.0 for s in estados}
    for _ in range(100):
        V_nuevo = {}
        for s in estados:
            valores = []
            for a in acciones:
                valor = sum(
                    prob * (mdp.obtener_recompensa(s, a, s_sig) + mdp.gamma * V[s_sig])
                    for s_sig, prob in mdp.obtener_transiciones(s, a)
                )
                valores.append(valor)
            V_nuevo[s] = max(valores) if valores else 0
        V = V_nuevo
    
    # Política
    politica = {}
    for s in estados:
        mejor_accion = None
        mejor_valor = float('-inf')
        for a in acciones:
            valor = sum(
                prob * (mdp.obtener_recompensa(s, a, s_sig) + mdp.gamma * V[s_sig])
                for s_sig, prob in mdp.obtener_transiciones(s, a)
            )
            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = a
        politica[s] = mejor_accion
    
    print("Política Óptima de Pedidos:")
    print("Inventario → Cantidad a pedir")
    for s in sorted(estados, key=int):
        print(f"  {s} → {politica[s]} unidades")
    
    print("\nFunción de Valor:")
    for s in sorted(estados, key=int):
        print(f"  V*({s}) = ${V[s]:.2f}")


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Proceso de Decisión de Markov (MDP) ===\n")
    
    ejemplo_estudiante()
    ejemplo_inventario()
    
    print("\n" + "="*70)
    print("\nComponentes de un MDP:")
    print("- S: Conjunto de estados")
    print("- A: Conjunto de acciones")
    print("- P(s'|s,a): Probabilidades de transición")
    print("- R(s,a,s'): Función de recompensa")
    print("- γ: Factor de descuento (0 ≤ γ < 1)")
    print("\nObjetivo:")
    print("- Encontrar política π* que maximiza el retorno esperado")
    print("- Retorno: Σ γ^t R_t")
    print("\nMétodos de solución:")
    print("- Iteración de valores")
    print("- Iteración de políticas")
    print("- Q-learning (aprendizaje por refuerzo)")

