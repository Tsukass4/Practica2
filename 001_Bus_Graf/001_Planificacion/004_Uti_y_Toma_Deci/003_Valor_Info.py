"""
Algoritmo 26: Valor de la Información (Value of Information)

El valor de la información cuantifica cuánto vale obtener información adicional
antes de tomar una decisión. Es la diferencia entre la utilidad esperada con
y sin la información.

VPI(E|e) = EU(α_E|e,E) - EU(α|e)

donde:
- E: Nueva evidencia/información
- e: Evidencia actual
- α: Mejor acción sin información adicional
- α_E: Mejor acción con información adicional
"""

from typing import Dict, List, Tuple
import itertools


class ProblemaDecision:
    """Problema de decisión simple para calcular VPI"""
    
    def __init__(self, estados: List[str], prob_estados: Dict[str, float],
                 acciones: List[str], utilidades: Dict[Tuple[str, str], float]):
        """
        Args:
            estados: Lista de posibles estados del mundo
            prob_estados: P(estado) para cada estado
            acciones: Lista de acciones posibles
            utilidades: U(estado, accion) para cada combinación
        """
        self.estados = estados
        self.prob_estados = prob_estados
        self.acciones = acciones
        self.utilidades = utilidades
    
    def utilidad_esperada_accion(self, accion: str, 
                                prob_estados: Dict[str, float] = None) -> float:
        """Calcula la utilidad esperada de una acción"""
        if prob_estados is None:
            prob_estados = self.prob_estados
        
        return sum(prob_estados.get(s, 0) * self.utilidades.get((s, accion), 0)
                  for s in self.estados)
    
    def mejor_accion(self, prob_estados: Dict[str, float] = None) -> Tuple[str, float]:
        """Encuentra la mejor acción y su utilidad esperada"""
        mejor_acc = None
        mejor_util = float('-inf')
        
        for accion in self.acciones:
            util = self.utilidad_esperada_accion(accion, prob_estados)
            if util > mejor_util:
                mejor_util = util
                mejor_acc = accion
        
        return mejor_acc, mejor_util
    
    def valor_informacion_perfecta(self, variable: str,
                                   prob_valores: Dict[str, float],
                                   prob_estado_dado_valor: Dict[Tuple[str, str], float]) -> float:
        """
        Calcula el valor de la información perfecta sobre una variable.
        
        Args:
            variable: Nombre de la variable
            prob_valores: P(valor) para cada valor de la variable
            prob_estado_dado_valor: P(estado|valor) para cada combinación
        
        Returns:
            VPI: Valor de la información perfecta
        """
        # Utilidad esperada sin información adicional
        _, eu_sin_info = self.mejor_accion()
        
        # Utilidad esperada con información perfecta
        eu_con_info = 0.0
        
        for valor in prob_valores.keys():
            # Actualizar creencias dado el valor observado
            prob_estados_actualizadas = {}
            for estado in self.estados:
                prob_estados_actualizadas[estado] = prob_estado_dado_valor.get((estado, valor), 0)
            
            # Normalizar
            suma = sum(prob_estados_actualizadas.values())
            if suma > 0:
                prob_estados_actualizadas = {s: p/suma for s, p in prob_estados_actualizadas.items()}
            
            # Mejor acción dado este valor
            _, util_dado_valor = self.mejor_accion(prob_estados_actualizadas)
            
            # Ponderar por probabilidad del valor
            eu_con_info += prob_valores[valor] * util_dado_valor
        
        # VPI = diferencia
        return eu_con_info - eu_sin_info


# Ejemplo 1: Problema del petróleo
def ejemplo_petroleo():
    """
    Problema: ¿Perforar en busca de petróleo?
    
    Estados: {petroleo, seco}
    Acciones: {perforar, no_perforar}
    Información: Resultado de prueba sísmica
    """
    print("=== Ejemplo: Problema del Petróleo ===\n")
    
    # Definir problema
    estados = ["petroleo", "seco"]
    prob_estados = {
        "petroleo": 0.5,
        "seco": 0.5
    }
    acciones = ["perforar", "no_perforar"]
    utilidades = {
        ("petroleo", "perforar"): 100,      # Encontrar petróleo
        ("petroleo", "no_perforar"): 0,     # Perder oportunidad
        ("seco", "perforar"): -70,          # Perforar en vano
        ("seco", "no_perforar"): 0          # No hacer nada
    }
    
    problema = ProblemaDecision(estados, prob_estados, acciones, utilidades)
    
    # Decisión sin información adicional
    print("Sin información adicional:")
    for accion in acciones:
        eu = problema.utilidad_esperada_accion(accion)
        print(f"  EU({accion}) = {eu:.2f}")
    
    mejor_acc, mejor_eu = problema.mejor_accion()
    print(f"\nMejor acción: {mejor_acc}")
    print(f"Utilidad esperada: {mejor_eu:.2f}")
    
    # Valor de información perfecta sobre el estado
    print("\n--- Valor de Información Perfecta ---")
    
    # Si supiéramos con certeza si hay petróleo
    prob_valores_test = {
        "positivo": 0.5,  # P(test positivo) = P(petróleo)
        "negativo": 0.5   # P(test negativo) = P(seco)
    }
    
    # P(estado | resultado test) - asumiendo test perfecto
    prob_estado_dado_test = {
        ("petroleo", "positivo"): 1.0,
        ("petroleo", "negativo"): 0.0,
        ("seco", "positivo"): 0.0,
        ("seco", "negativo"): 1.0
    }
    
    vpi = problema.valor_informacion_perfecta("test", prob_valores_test, prob_estado_dado_test)
    print(f"VPI (test perfecto) = ${vpi:.2f}")
    print(f"\nInterpretación: Pagaríamos hasta ${vpi:.2f} por un test perfecto")
    
    # Valor con test imperfecto
    print("\n--- Valor de Información Imperfecta ---")
    
    # Test con 80% de precisión
    prob_estado_dado_test_imperfecto = {
        ("petroleo", "positivo"): 0.8,
        ("petroleo", "negativo"): 0.2,
        ("seco", "positivo"): 0.2,
        ("seco", "negativo"): 0.8
    }
    
    # Calcular P(resultado test)
    prob_test = {}
    for resultado in ["positivo", "negativo"]:
        prob_test[resultado] = sum(
            prob_estados[estado] * prob_estado_dado_test_imperfecto.get((estado, resultado), 0)
            for estado in estados
        )
    
    vpi_imperfecto = problema.valor_informacion_perfecta("test", prob_test, prob_estado_dado_test_imperfecto)
    print(f"VPI (test 80% preciso) = ${vpi_imperfecto:.2f}")
    print(f"\nInterpretación: Pagaríamos hasta ${vpi_imperfecto:.2f} por un test 80% preciso")


# Ejemplo 2: Problema médico
def ejemplo_medico_vpi():
    """
    Problema médico: ¿Realizar tratamiento?
    ¿Vale la pena hacer una prueba diagnóstica?
    """
    print("\n\n=== Ejemplo: Decisión Médica con VPI ===\n")
    
    # Estados: enfermedad presente o ausente
    estados = ["enfermo", "sano"]
    prob_estados = {
        "enfermo": 0.1,  # 10% probabilidad a priori
        "sano": 0.9
    }
    
    # Acciones: tratar o no tratar
    acciones = ["tratar", "no_tratar"]
    utilidades = {
        ("enfermo", "tratar"): 80,       # Curado
        ("enfermo", "no_tratar"): 0,     # Enfermo sin tratar
        ("sano", "tratar"): 40,          # Efectos secundarios innecesarios
        ("sano", "no_tratar"): 100       # Sano sin tratamiento
    }
    
    problema = ProblemaDecision(estados, prob_estados, acciones, utilidades)
    
    # Decisión sin prueba
    print("Sin prueba diagnóstica:")
    for accion in acciones:
        eu = problema.utilidad_esperada_accion(accion)
        print(f"  EU({accion}) = {eu:.2f}")
    
    mejor_acc, mejor_eu = problema.mejor_accion()
    print(f"\nMejor acción: {mejor_acc}")
    print(f"Utilidad esperada: {mejor_eu:.2f}")
    
    # Valor de prueba diagnóstica (90% sensibilidad, 95% especificidad)
    print("\n--- Valor de Prueba Diagnóstica ---")
    
    sensibilidad = 0.90  # P(positivo | enfermo)
    especificidad = 0.95  # P(negativo | sano)
    
    prob_estado_dado_resultado = {
        ("enfermo", "positivo"): sensibilidad,
        ("enfermo", "negativo"): 1 - sensibilidad,
        ("sano", "positivo"): 1 - especificidad,
        ("sano", "negativo"): especificidad
    }
    
    # Calcular P(resultado)
    prob_resultado = {}
    for resultado in ["positivo", "negativo"]:
        prob_resultado[resultado] = sum(
            prob_estados[estado] * prob_estado_dado_resultado.get((estado, resultado), 0)
            for estado in estados
        )
    
    vpi = problema.valor_informacion_perfecta("prueba", prob_resultado, prob_estado_dado_resultado)
    print(f"Sensibilidad: {sensibilidad*100}%")
    print(f"Especificidad: {especificidad*100}%")
    print(f"\nVPI (prueba diagnóstica) = {vpi:.2f} unidades de utilidad")
    
    costo_prueba = 5  # En unidades de utilidad
    print(f"Costo de la prueba: {costo_prueba}")
    
    if vpi > costo_prueba:
        print(f"\n✓ Vale la pena hacer la prueba (VPI > costo)")
        print(f"  Beneficio neto: {vpi - costo_prueba:.2f}")
    else:
        print(f"\n✗ No vale la pena hacer la prueba (VPI < costo)")


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Valor de la Información (VPI) ===\n")
    
    ejemplo_petroleo()
    ejemplo_medico_vpi()
    
    print("\n" + "="*70)
    print("\nConceptos clave:")
    print("- VPI: Valor de la Información Perfecta")
    print("- VPI ≥ 0 siempre (información nunca perjudica)")
    print("- VPI = 0 si la información no cambia la decisión")
    print("- Información imperfecta tiene menor valor que perfecta")
    print("\nAplicaciones:")
    print("- Decidir si realizar pruebas diagnósticas")
    print("- Evaluar estudios de mercado")
    print("- Análisis de riesgo en inversiones")
    print("- Exploración de recursos naturales")

