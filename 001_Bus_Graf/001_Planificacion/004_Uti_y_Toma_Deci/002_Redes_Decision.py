"""
Algoritmo 25: Redes de Decisión (Decision Networks / Influence Diagrams)

Las redes de decisión extienden las redes bayesianas con nodos de decisión
y utilidad. Permiten modelar problemas de decisión secuencial bajo incertidumbre.

Componentes:
- Nodos de azar (chance): Variables aleatorias
- Nodos de decisión: Acciones que puede tomar el agente
- Nodos de utilidad: Función de utilidad
- Arcos: Dependencias probabilísticas e informacionales
"""

from typing import Dict, List, Tuple, Set, Optional
import itertools


class NodoAzar:
    """Nodo de variable aleatoria"""
    
    def __init__(self, nombre: str, valores: List[str], 
                 padres: List[str], tabla_prob: Dict):
        self.nombre = nombre
        self.valores = valores
        self.padres = padres
        self.tabla_prob = tabla_prob  # CPT: Conditional Probability Table
    
    def probabilidad(self, valor: str, evidencia: Dict[str, str]) -> float:
        """Calcula P(valor | evidencia)"""
        if not self.padres:
            return self.tabla_prob.get(valor, 0.0)
        
        # Construir clave para la tabla
        clave_padres = tuple(evidencia.get(p, None) for p in self.padres)
        return self.tabla_prob.get((clave_padres, valor), 0.0)


class NodoDecision:
    """Nodo de decisión"""
    
    def __init__(self, nombre: str, opciones: List[str], padres: List[str] = None):
        self.nombre = nombre
        self.opciones = opciones
        self.padres = padres or []  # Información disponible al decidir


class NodoUtilidad:
    """Nodo de utilidad"""
    
    def __init__(self, nombre: str, padres: List[str], tabla_utilidad: Dict):
        self.nombre = nombre
        self.padres = padres
        self.tabla_utilidad = tabla_utilidad
    
    def utilidad(self, valores_padres: Dict[str, str]) -> float:
        """Calcula la utilidad dados los valores de los padres"""
        clave = tuple(valores_padres.get(p, None) for p in self.padres)
        return self.tabla_utilidad.get(clave, 0.0)


class RedDecision:
    """Red de decisión (diagrama de influencia)"""
    
    def __init__(self):
        self.nodos_azar: Dict[str, NodoAzar] = {}
        self.nodos_decision: Dict[str, NodoDecision] = {}
        self.nodos_utilidad: Dict[str, NodoUtilidad] = {}
    
    def agregar_nodo_azar(self, nodo: NodoAzar):
        """Agrega un nodo de azar"""
        self.nodos_azar[nodo.nombre] = nodo
    
    def agregar_nodo_decision(self, nodo: NodoDecision):
        """Agrega un nodo de decisión"""
        self.nodos_decision[nodo.nombre] = nodo
    
    def agregar_nodo_utilidad(self, nodo: NodoUtilidad):
        """Agrega un nodo de utilidad"""
        self.nodos_utilidad[nodo.nombre] = nodo
    
    def utilidad_esperada(self, decision: Dict[str, str], 
                         evidencia: Dict[str, str] = None) -> float:
        """
        Calcula la utilidad esperada de una decisión.
        
        Args:
            decision: Asignación de valores a nodos de decisión
            evidencia: Evidencia observada
        
        Returns:
            Utilidad esperada
        """
        if evidencia is None:
            evidencia = {}
        
        # Combinar decisión y evidencia
        asignacion_fija = {**evidencia, **decision}
        
        # Obtener variables aleatorias no observadas
        vars_aleatorias = [v for v in self.nodos_azar.keys() 
                          if v not in asignacion_fija]
        
        # Sumar sobre todas las asignaciones posibles
        utilidad_total = 0.0
        
        for asignacion_vars in self._generar_asignaciones(vars_aleatorias):
            asignacion_completa = {**asignacion_fija, **asignacion_vars}
            
            # Calcular probabilidad de esta asignación
            prob = self._probabilidad_conjunta(asignacion_completa)
            
            # Calcular utilidad de esta asignación
            util = sum(nodo.utilidad(asignacion_completa) 
                      for nodo in self.nodos_utilidad.values())
            
            utilidad_total += prob * util
        
        return utilidad_total
    
    def mejor_decision(self, evidencia: Dict[str, str] = None) -> Tuple[Dict[str, str], float]:
        """
        Encuentra la mejor decisión que maximiza la utilidad esperada.
        
        Returns:
            Tupla (mejor_decision, utilidad_esperada)
        """
        if evidencia is None:
            evidencia = {}
        
        mejor_decision = None
        mejor_utilidad = float('-inf')
        
        # Generar todas las combinaciones de decisiones
        nombres_decision = list(self.nodos_decision.keys())
        opciones_decision = [self.nodos_decision[n].opciones for n in nombres_decision]
        
        for combinacion in itertools.product(*opciones_decision):
            decision = dict(zip(nombres_decision, combinacion))
            utilidad = self.utilidad_esperada(decision, evidencia)
            
            if utilidad > mejor_utilidad:
                mejor_utilidad = utilidad
                mejor_decision = decision
        
        return mejor_decision, mejor_utilidad
    
    def _generar_asignaciones(self, variables: List[str]) -> List[Dict[str, str]]:
        """Genera todas las asignaciones posibles para las variables"""
        if not variables:
            return [{}]
        
        asignaciones = []
        valores_vars = [self.nodos_azar[v].valores for v in variables]
        
        for combinacion in itertools.product(*valores_vars):
            asignaciones.append(dict(zip(variables, combinacion)))
        
        return asignaciones
    
    def _probabilidad_conjunta(self, asignacion: Dict[str, str]) -> float:
        """Calcula la probabilidad conjunta de una asignación"""
        prob = 1.0
        
        for nombre, nodo in self.nodos_azar.items():
            if nombre in asignacion:
                prob *= nodo.probabilidad(asignacion[nombre], asignacion)
        
        return prob


# Ejemplo de uso: Problema del paraguas
def ejemplo_paraguas():
    """
    Problema de decisión: ¿Llevar paraguas?
    
    Variables:
    - Clima: {soleado, lluvioso}
    - Decisión: {llevar_paraguas, no_llevar}
    - Utilidad: Depende del clima y la decisión
    """
    print("=== Ejemplo: Problema del Paraguas ===\n")
    
    red = RedDecision()
    
    # Nodo de azar: Clima
    clima = NodoAzar(
        nombre="Clima",
        valores=["soleado", "lluvioso"],
        padres=[],
        tabla_prob={
            "soleado": 0.7,
            "lluvioso": 0.3
        }
    )
    red.agregar_nodo_azar(clima)
    
    # Nodo de decisión: Llevar paraguas
    decision_paraguas = NodoDecision(
        nombre="Paraguas",
        opciones=["llevar", "no_llevar"],
        padres=[]  # No tiene información del clima al decidir
    )
    red.agregar_nodo_decision(decision_paraguas)
    
    # Nodo de utilidad
    utilidad = NodoUtilidad(
        nombre="Utilidad",
        padres=["Clima", "Paraguas"],
        tabla_utilidad={
            ("soleado", "llevar"): 20,      # Molestia de cargar paraguas
            ("soleado", "no_llevar"): 100,  # Perfecto
            ("lluvioso", "llevar"): 70,     # Seco pero con paraguas
            ("lluvioso", "no_llevar"): 0    # Mojado
        }
    )
    red.agregar_nodo_utilidad(utilidad)
    
    # Evaluar decisiones
    print("Utilidades esperadas:")
    for opcion in decision_paraguas.opciones:
        decision = {"Paraguas": opcion}
        ue = red.utilidad_esperada(decision)
        print(f"  {opcion}: {ue:.2f}")
    
    # Mejor decisión
    mejor_dec, mejor_util = red.mejor_decision()
    print(f"\nMejor decisión: {mejor_dec}")
    print(f"Utilidad esperada: {mejor_util:.2f}")
    
    # Con evidencia (sabemos que va a llover)
    print("\n--- Con evidencia: Clima = lluvioso ---")
    mejor_dec_ev, mejor_util_ev = red.mejor_decision({"Clima": "lluvioso"})
    print(f"Mejor decisión: {mejor_dec_ev}")
    print(f"Utilidad esperada: {mejor_util_ev:.2f}")


# Ejemplo de uso: Problema médico
def ejemplo_medico():
    """
    Problema de decisión médica: ¿Realizar prueba y tratamiento?
    
    Variables:
    - Enfermedad: {presente, ausente}
    - Decisión_Prueba: {hacer_prueba, no_hacer}
    - Resultado_Prueba: {positivo, negativo} (si se hace)
    - Decisión_Tratamiento: {tratar, no_tratar}
    - Utilidad: Depende de enfermedad, tratamiento y costo de prueba
    """
    print("\n\n=== Ejemplo: Decisión Médica ===\n")
    
    red = RedDecision()
    
    # Nodo de azar: Enfermedad
    enfermedad = NodoAzar(
        nombre="Enfermedad",
        valores=["presente", "ausente"],
        padres=[],
        tabla_prob={
            "presente": 0.1,
            "ausente": 0.9
        }
    )
    red.agregar_nodo_azar(enfermedad)
    
    # Decisión: Tratamiento
    decision_tratamiento = NodoDecision(
        nombre="Tratamiento",
        opciones=["tratar", "no_tratar"],
        padres=["Enfermedad"]  # Simplificado: asumimos conocemos el estado
    )
    red.agregar_nodo_decision(decision_tratamiento)
    
    # Utilidad
    utilidad = NodoUtilidad(
        nombre="Utilidad",
        padres=["Enfermedad", "Tratamiento"],
        tabla_utilidad={
            ("presente", "tratar"): 80,      # Curado, pero costo tratamiento
            ("presente", "no_tratar"): 0,    # Enfermo
            ("ausente", "tratar"): 40,       # Sano pero efectos secundarios
            ("ausente", "no_tratar"): 100    # Sano y sin tratamiento
        }
    )
    red.agregar_nodo_utilidad(utilidad)
    
    # Evaluar decisiones
    print("Utilidades esperadas:")
    for opcion in decision_tratamiento.opciones:
        decision = {"Tratamiento": opcion}
        ue = red.utilidad_esperada(decision)
        print(f"  {opcion}: {ue:.2f}")
    
    # Mejor decisión sin información
    mejor_dec, mejor_util = red.mejor_decision()
    print(f"\nMejor decisión (sin información): {mejor_dec}")
    print(f"Utilidad esperada: {mejor_util:.2f}")
    
    # Con evidencia (sabemos que hay enfermedad)
    print("\n--- Con evidencia: Enfermedad = presente ---")
    mejor_dec_ev, mejor_util_ev = red.mejor_decision({"Enfermedad": "presente"})
    print(f"Mejor decisión: {mejor_dec_ev}")
    print(f"Utilidad esperada: {mejor_util_ev:.2f}")


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Redes de Decisión (Decision Networks) ===\n")
    
    ejemplo_paraguas()
    ejemplo_medico()
    
    print("\n" + "="*70)
    print("\nCaracterísticas de las Redes de Decisión:")
    print("- Extienden redes bayesianas con decisiones y utilidad")
    print("- Modelan decisiones secuenciales bajo incertidumbre")
    print("- Permiten calcular utilidad esperada de decisiones")
    print("- Útiles para análisis de decisiones complejas")
    print("\nComponentes:")
    print("- Nodos de azar: Variables aleatorias")
    print("- Nodos de decisión: Acciones del agente")
    print("- Nodos de utilidad: Función de utilidad")

