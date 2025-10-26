"""
Algoritmo 38: Incertidumbre

La incertidumbre es fundamental en IA cuando no tenemos información completa
sobre el mundo. La teoría de probabilidad nos permite razonar bajo incertidumbre.

Conceptos clave:
- Incertidumbre epistémica (falta de conocimiento)
- Incertidumbre aleatoria (variabilidad inherente)
- Modelado probabilístico
- Toma de decisiones bajo incertidumbre
"""

import random
from typing import Dict, List, Tuple
from collections import defaultdict


class ModeloIncertidumbre:
    """Modelo para representar y manejar incertidumbre"""
    
    def __init__(self):
        self.eventos = {}
        self.observaciones = []
    
    def agregar_evento(self, nombre: str, probabilidad: float):
        """Agrega un evento con su probabilidad"""
        if 0 <= probabilidad <= 1:
            self.eventos[nombre] = probabilidad
        else:
            raise ValueError("La probabilidad debe estar entre 0 y 1")
    
    def simular_evento(self, nombre: str) -> bool:
        """Simula si un evento ocurre basado en su probabilidad"""
        if nombre not in self.eventos:
            raise ValueError(f"Evento {nombre} no definido")
        return random.random() < self.eventos[nombre]
    
    def experimento_montecarlo(self, nombre: str, n_simulaciones: int = 1000) -> float:
        """Estima la probabilidad de un evento mediante simulación"""
        exitos = sum(1 for _ in range(n_simulaciones) if self.simular_evento(nombre))
        return exitos / n_simulaciones


class SistemaExperto:
    """Sistema experto simple que maneja incertidumbre"""
    
    def __init__(self):
        self.reglas = []  # (condiciones, conclusion, certeza)
        self.hechos = {}  # hecho -> certeza
    
    def agregar_regla(self, condiciones: List[str], conclusion: str, certeza: float):
        """Agrega una regla con factor de certeza"""
        self.reglas.append((condiciones, conclusion, certeza))
    
    def agregar_hecho(self, hecho: str, certeza: float = 1.0):
        """Agrega un hecho con su certeza"""
        self.hechos[hecho] = certeza
    
    def inferir(self):
        """Realiza inferencia propagando certezas"""
        cambios = True
        while cambios:
            cambios = False
            for condiciones, conclusion, certeza_regla in self.reglas:
                # Verificar si todas las condiciones se cumplen
                certezas_condiciones = []
                for cond in condiciones:
                    if cond in self.hechos:
                        certezas_condiciones.append(self.hechos[cond])
                    else:
                        break
                else:
                    # Todas las condiciones presentes
                    # Certeza de la conclusión = min(certezas_condiciones) * certeza_regla
                    certeza_conclusion = min(certezas_condiciones) * certeza_regla
                    
                    if conclusion not in self.hechos or self.hechos[conclusion] < certeza_conclusion:
                        self.hechos[conclusion] = certeza_conclusion
                        cambios = True


# Ejemplo 1: Diagnóstico médico con incertidumbre
def ejemplo_diagnostico_medico():
    """Ejemplo de diagnóstico médico bajo incertidumbre"""
    print("=== Diagnóstico Médico con Incertidumbre ===\n")
    
    sistema = SistemaExperto()
    
    # Reglas con factores de certeza
    sistema.agregar_regla(["fiebre", "tos"], "gripe", 0.7)
    sistema.agregar_regla(["fiebre", "dolor_cabeza"], "gripe", 0.6)
    sistema.agregar_regla(["fiebre", "tos", "fatiga"], "gripe", 0.9)
    sistema.agregar_regla(["fiebre", "erupcion"], "sarampion", 0.8)
    sistema.agregar_regla(["tos", "dificultad_respirar"], "neumonia", 0.75)
    
    # Observaciones del paciente
    print("Síntomas observados:")
    sistema.agregar_hecho("fiebre", 0.9)
    sistema.agregar_hecho("tos", 0.8)
    sistema.agregar_hecho("fatiga", 0.7)
    
    print(f"  - Fiebre (certeza: 0.9)")
    print(f"  - Tos (certeza: 0.8)")
    print(f"  - Fatiga (certeza: 0.7)")
    
    # Inferir diagnósticos
    sistema.inferir()
    
    print("\nDiagnósticos inferidos:")
    diagnosticos = [(h, c) for h, c in sistema.hechos.items() 
                   if h not in ["fiebre", "tos", "fatiga", "dolor_cabeza", "erupcion", "dificultad_respirar"]]
    diagnosticos.sort(key=lambda x: x[1], reverse=True)
    
    for diagnostico, certeza in diagnosticos:
        print(f"  - {diagnostico}: {certeza:.2f} ({certeza*100:.0f}% de certeza)")


# Ejemplo 2: Predicción del clima
def ejemplo_prediccion_clima():
    """Ejemplo de predicción del clima con incertidumbre"""
    print("\n\n=== Predicción del Clima con Incertidumbre ===\n")
    
    modelo = ModeloIncertidumbre()
    
    # Probabilidades basadas en datos históricos
    modelo.agregar_evento("lluvia_manana", 0.3)
    modelo.agregar_evento("nublado_hoy", 0.6)
    modelo.agregar_evento("viento_fuerte", 0.2)
    
    print("Probabilidades basadas en datos históricos:")
    print(f"  - P(Lluvia mañana) = 0.30")
    print(f"  - P(Nublado hoy) = 0.60")
    print(f"  - P(Viento fuerte) = 0.20")
    
    # Simulación Monte Carlo
    print("\nSimulación de 1000 días:")
    prob_estimada = modelo.experimento_montecarlo("lluvia_manana", 1000)
    print(f"  - Probabilidad estimada de lluvia: {prob_estimada:.3f}")
    
    # Simular una semana
    print("\nSimulación de una semana:")
    dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    
    for dia in dias:
        llueve = modelo.simular_evento("lluvia_manana")
        nublado = modelo.simular_evento("nublado_hoy")
        viento = modelo.simular_evento("viento_fuerte")
        
        condiciones = []
        if llueve:
            condiciones.append("Lluvia")
        if nublado:
            condiciones.append("Nublado")
        if viento:
            condiciones.append("Viento")
        
        if not condiciones:
            condiciones.append("Despejado")
        
        print(f"  {dia}: {', '.join(condiciones)}")


# Ejemplo 3: Juego de azar
def ejemplo_juego_azar():
    """Ejemplo de toma de decisiones en juego con incertidumbre"""
    print("\n\n=== Juego de Azar: Decisión bajo Incertidumbre ===\n")
    
    print("Juego: Lanzar un dado")
    print("  - Si sale 6: Ganas $100")
    print("  - Si sale 1-5: Pierdes $10")
    print("  - Costo de jugar: $5")
    
    # Calcular valor esperado
    prob_ganar = 1/6
    prob_perder = 5/6
    ganancia_ganar = 100 - 5
    ganancia_perder = -10 - 5
    
    valor_esperado = prob_ganar * ganancia_ganar + prob_perder * ganancia_perder
    
    print(f"\nAnálisis:")
    print(f"  P(Ganar) = {prob_ganar:.3f}")
    print(f"  P(Perder) = {prob_perder:.3f}")
    print(f"  Ganancia si ganas = ${ganancia_ganar}")
    print(f"  Pérdida si pierdes = ${ganancia_perder}")
    print(f"\n  Valor Esperado = {valor_esperado:.2f}")
    
    if valor_esperado > 0:
        print(f"\n  Decisión: JUGAR (valor esperado positivo)")
    else:
        print(f"\n  Decisión: NO JUGAR (valor esperado negativo)")
    
    # Simular 100 juegos
    print("\nSimulación de 100 juegos:")
    balance = 0
    for _ in range(100):
        resultado = random.randint(1, 6)
        if resultado == 6:
            balance += ganancia_ganar
        else:
            balance += ganancia_perder
    
    print(f"  Balance final: ${balance}")
    print(f"  Promedio por juego: ${balance/100:.2f}")


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Incertidumbre en Inteligencia Artificial ===\n")
    
    ejemplo_diagnostico_medico()
    ejemplo_prediccion_clima()
    ejemplo_juego_azar()
    
    print("\n" + "="*70)
    print("\nTipos de Incertidumbre:")
    print("  1. Incertidumbre epistémica: Falta de conocimiento")
    print("  2. Incertidumbre aleatoria: Variabilidad inherente del sistema")
    print("\nFormas de Manejar Incertidumbre:")
    print("  - Teoría de probabilidad")
    print("  - Lógica difusa")
    print("  - Factores de certeza")
    print("  - Redes bayesianas")
    print("\nAplicaciones:")
    print("  - Diagnóstico médico")
    print("  - Predicción del clima")
    print("  - Toma de decisiones financieras")
    print("  - Sistemas expertos")

