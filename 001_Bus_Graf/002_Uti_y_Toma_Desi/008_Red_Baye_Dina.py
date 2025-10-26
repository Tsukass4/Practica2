"""
Algoritmo 31: Red Bayesiana Dinámica (Dynamic Bayesian Network - DBN)

Una DBN es una red bayesiana que modela procesos temporales.
Extiende las redes bayesianas estáticas para representar cómo
las variables evolucionan en el tiempo.

Componentes:
- Variables en cada paso de tiempo t
- Modelo de transición: P(X_t | X_{t-1})
- Modelo de observación: P(E_t | X_t)
- Inferencia temporal (filtrado, predicción, suavizado)
"""

from typing import Dict, List, Tuple, Optional
import random


class RedBayesianaDinamica:
    """Red Bayesiana Dinámica simple"""
    
    def __init__(self, variables_estado: List[str], variables_observacion: List[str],
                 valores_posibles: Dict[str, List[str]],
                 prob_inicial: Dict[str, Dict[str, float]],
                 modelo_transicion: Dict[Tuple[str, str, str], float],
                 modelo_observacion: Dict[Tuple[str, str, str], float]):
        """
        Args:
            variables_estado: Variables de estado (ocultas)
            variables_observacion: Variables observables
            valores_posibles: Valores posibles para cada variable
            prob_inicial: P(X_0) distribución inicial
            modelo_transicion: P(X_t | X_{t-1})
            modelo_observacion: P(E_t | X_t)
        """
        self.variables_estado = variables_estado
        self.variables_observacion = variables_observacion
        self.valores_posibles = valores_posibles
        self.prob_inicial = prob_inicial
        self.modelo_transicion = modelo_transicion
        self.modelo_observacion = modelo_observacion
        
        # Estado de creencia actual
        self.creencia = prob_inicial.copy()
    
    def predecir(self, creencia: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Paso de predicción: P(X_{t+1} | e_{1:t})
        
        P(X_{t+1} | e_{1:t}) = Σ_{x_t} P(X_{t+1} | x_t) P(x_t | e_{1:t})
        """
        nueva_creencia = {}
        
        for var in self.variables_estado:
            nueva_creencia[var] = {}
            
            for valor_nuevo in self.valores_posibles[var]:
                prob = 0.0
                
                # Sumar sobre todos los valores anteriores
                for valor_viejo in self.valores_posibles[var]:
                    prob_transicion = self.modelo_transicion.get(
                        (var, valor_viejo, valor_nuevo), 0.0
                    )
                    prob_anterior = creencia[var].get(valor_viejo, 0.0)
                    prob += prob_transicion * prob_anterior
                
                nueva_creencia[var][valor_nuevo] = prob
        
        return nueva_creencia
    
    def actualizar(self, creencia: Dict[str, Dict[str, float]], 
                   evidencia: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Paso de actualización: P(X_t | e_{1:t})
        
        P(X_t | e_{1:t}) = α P(e_t | X_t) P(X_t | e_{1:t-1})
        """
        nueva_creencia = {}
        
        for var in self.variables_estado:
            nueva_creencia[var] = {}
            
            for valor in self.valores_posibles[var]:
                # Probabilidad a priori (de la predicción)
                prob_prior = creencia[var].get(valor, 0.0)
                
                # Likelihood: P(evidencia | estado)
                likelihood = 1.0
                for var_obs, valor_obs in evidencia.items():
                    prob_obs = self.modelo_observacion.get(
                        (var, valor, valor_obs), 1.0
                    )
                    likelihood *= prob_obs
                
                nueva_creencia[var][valor] = prob_prior * likelihood
        
        # Normalizar
        for var in self.variables_estado:
            total = sum(nueva_creencia[var].values())
            if total > 0:
                nueva_creencia[var] = {v: p/total for v, p in nueva_creencia[var].items()}
        
        return nueva_creencia
    
    def filtrar(self, evidencia: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Filtrado: Actualiza la creencia con nueva evidencia.
        
        Combina predicción y actualización.
        """
        # Predecir
        creencia_predicha = self.predecir(self.creencia)
        
        # Actualizar con evidencia
        self.creencia = self.actualizar(creencia_predicha, evidencia)
        
        return self.creencia
    
    def estado_mas_probable(self) -> Dict[str, str]:
        """Retorna el estado más probable según la creencia actual"""
        estado = {}
        for var in self.variables_estado:
            mejor_valor = max(self.creencia[var].items(), key=lambda x: x[1])
            estado[var] = mejor_valor[0]
        return estado


# Ejemplo 1: Modelo de lluvia y paraguas
def ejemplo_lluvia_paraguas():
    """
    Modelo clásico de lluvia y paraguas:
    - Estado oculto: Lluvia (sí/no)
    - Observación: Paraguas (sí/no)
    
    Queremos inferir si está lloviendo basándonos en si la gente lleva paraguas.
    """
    print("=== Ejemplo: Modelo de Lluvia y Paraguas ===\n")
    
    variables_estado = ["lluvia"]
    variables_observacion = ["paraguas"]
    valores_posibles = {
        "lluvia": ["si", "no"],
        "paraguas": ["si", "no"]
    }
    
    # Distribución inicial: P(Lluvia_0)
    prob_inicial = {
        "lluvia": {"si": 0.5, "no": 0.5}
    }
    
    # Modelo de transición: P(Lluvia_t | Lluvia_{t-1})
    modelo_transicion = {
        ("lluvia", "si", "si"): 0.7,   # Si llueve, prob 0.7 de que siga lloviendo
        ("lluvia", "si", "no"): 0.3,
        ("lluvia", "no", "si"): 0.3,   # Si no llueve, prob 0.3 de que empiece
        ("lluvia", "no", "no"): 0.7,
    }
    
    # Modelo de observación: P(Paraguas | Lluvia)
    modelo_observacion = {
        ("lluvia", "si", "si"): 0.9,   # Si llueve, prob 0.9 de ver paraguas
        ("lluvia", "si", "no"): 0.1,
        ("lluvia", "no", "si"): 0.2,   # Si no llueve, prob 0.2 de ver paraguas
        ("lluvia", "no", "no"): 0.8,
    }
    
    dbn = RedBayesianaDinamica(
        variables_estado, variables_observacion, valores_posibles,
        prob_inicial, modelo_transicion, modelo_observacion
    )
    
    # Secuencia de observaciones
    observaciones = [
        {"paraguas": "si"},
        {"paraguas": "si"},
        {"paraguas": "no"},
        {"paraguas": "si"},
        {"paraguas": "si"}
    ]
    
    print("Secuencia de observaciones y creencias:\n")
    
    for t, obs in enumerate(observaciones):
        creencia = dbn.filtrar(obs)
        estado_probable = dbn.estado_mas_probable()
        
        print(f"Tiempo {t + 1}:")
        print(f"  Observación: Paraguas = {obs['paraguas']}")
        print(f"  P(Lluvia=sí) = {creencia['lluvia']['si']:.3f}")
        print(f"  P(Lluvia=no) = {creencia['lluvia']['no']:.3f}")
        print(f"  Estado más probable: Lluvia = {estado_probable['lluvia']}")
        print()


# Ejemplo 2: Seguimiento de robot
def ejemplo_seguimiento_robot():
    """
    Robot moviéndose en una línea:
    - Estado: Posición {0, 1, 2, 3, 4}
    - Observación: Sensor de distancia (ruidoso)
    """
    print("\n=== Ejemplo: Seguimiento de Robot ===\n")
    
    posiciones = ["0", "1", "2", "3", "4"]
    
    variables_estado = ["posicion"]
    variables_observacion = ["sensor"]
    valores_posibles = {
        "posicion": posiciones,
        "sensor": posiciones
    }
    
    # Distribución inicial (uniforme)
    prob_inicial = {
        "posicion": {p: 1.0/len(posiciones) for p in posiciones}
    }
    
    # Modelo de transición: robot se mueve a la derecha con ruido
    modelo_transicion = {}
    for i, pos in enumerate(posiciones):
        # 70% avanza, 20% se queda, 10% retrocede
        if i < len(posiciones) - 1:
            modelo_transicion[("posicion", pos, posiciones[i+1])] = 0.7
        else:
            modelo_transicion[("posicion", pos, pos)] = 0.7
        
        modelo_transicion[("posicion", pos, pos)] = modelo_transicion.get(("posicion", pos, pos), 0) + 0.2
        
        if i > 0:
            modelo_transicion[("posicion", pos, posiciones[i-1])] = 0.1
        else:
            modelo_transicion[("posicion", pos, pos)] = modelo_transicion.get(("posicion", pos, pos), 0) + 0.1
    
    # Modelo de observación: sensor con ruido gaussiano discretizado
    modelo_observacion = {}
    for pos_real in posiciones:
        for pos_sensor in posiciones:
            distancia = abs(int(pos_real) - int(pos_sensor))
            if distancia == 0:
                prob = 0.6
            elif distancia == 1:
                prob = 0.2
            elif distancia == 2:
                prob = 0.1
            else:
                prob = 0.05
            
            modelo_observacion[("posicion", pos_real, pos_sensor)] = prob
    
    dbn = RedBayesianaDinamica(
        variables_estado, variables_observacion, valores_posibles,
        prob_inicial, modelo_transicion, modelo_observacion
    )
    
    # Simular movimiento y observaciones
    observaciones = [
        {"sensor": "1"},
        {"sensor": "2"},
        {"sensor": "2"},
        {"sensor": "3"},
        {"sensor": "4"}
    ]
    
    print("Seguimiento de posición del robot:\n")
    
    for t, obs in enumerate(observaciones):
        creencia = dbn.filtrar(obs)
        estado_probable = dbn.estado_mas_probable()
        
        print(f"Tiempo {t + 1}:")
        print(f"  Sensor lee: {obs['sensor']}")
        print(f"  Distribución de creencia:")
        for pos in posiciones:
            prob = creencia['posicion'][pos]
            barra = '█' * int(prob * 20)
            print(f"    Pos {pos}: {prob:.3f} {barra}")
        print(f"  Posición más probable: {estado_probable['posicion']}")
        print()


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Red Bayesiana Dinámica (DBN) ===\n")
    
    ejemplo_lluvia_paraguas()
    ejemplo_seguimiento_robot()
    
    print("="*70)
    print("\nCaracterísticas de DBNs:")
    print("- Modelan procesos temporales estocásticos")
    print("- Extienden redes bayesianas al dominio temporal")
    print("- Permiten inferencia temporal (filtrado, predicción, suavizado)")
    print("\nComponentes:")
    print("- Modelo de transición: P(X_t | X_{t-1})")
    print("- Modelo de observación: P(E_t | X_t)")
    print("- Distribución inicial: P(X_0)")
    print("\nTipos de inferencia:")
    print("- Filtrado: P(X_t | e_{1:t}) - estado actual")
    print("- Predicción: P(X_{t+k} | e_{1:t}) - estado futuro")
    print("- Suavizado: P(X_k | e_{1:t}) - estado pasado")
    print("\nAplicaciones:")
    print("- Seguimiento de objetos")
    print("- Reconocimiento de voz")
    print("- Localización de robots")
    print("- Predicción de series temporales")

