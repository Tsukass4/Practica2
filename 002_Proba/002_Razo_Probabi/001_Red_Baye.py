"""
Algoritmo 44: Red Bayesiana

Una Red Bayesiana es un grafo acíclico dirigido donde:
- Nodos representan variables aleatorias
- Aristas representan dependencias probabilísticas
- Cada nodo tiene una tabla de probabilidad condicional (CPT)

Ventajas:
- Representación compacta de distribuciones conjuntas
- Aprovecha independencias condicionales
- Permite inferencia eficiente
"""

from typing import Dict, List, Tuple, Set
from collections import defaultdict
import itertools


class NodoRedBayesiana:
    """Nodo en una red bayesiana"""
    
    def __init__(self, nombre: str, valores_posibles: List[str]):
        self.nombre = nombre
        self.valores_posibles = valores_posibles
        self.padres = []
        # CPT: Tabla de Probabilidad Condicional
        # {(valor_padre1, valor_padre2, ...): {valor_nodo: probabilidad}}
        self.cpt = {}
    
    def agregar_padre(self, padre: 'NodoRedBayesiana'):
        """Agrega un nodo padre"""
        if padre not in self.padres:
            self.padres.append(padre)
    
    def establecer_probabilidad(self, valores_padres: Tuple, valor_nodo: str, probabilidad: float):
        """
        Establece P(nodo=valor_nodo | padres=valores_padres)
        
        Args:
            valores_padres: Tupla con valores de los padres (vacía si no hay padres)
            valor_nodo: Valor de este nodo
            probabilidad: P(nodo | padres)
        """
        if valores_padres not in self.cpt:
            self.cpt[valores_padres] = {}
        self.cpt[valores_padres][valor_nodo] = probabilidad
    
    def obtener_probabilidad(self, valores_padres: Tuple, valor_nodo: str) -> float:
        """Obtiene P(nodo=valor_nodo | padres=valores_padres)"""
        return self.cpt.get(valores_padres, {}).get(valor_nodo, 0.0)


class RedBayesiana:
    """Red Bayesiana completa"""
    
    def __init__(self):
        self.nodos = {}
    
    def agregar_nodo(self, nodo: NodoRedBayesiana):
        """Agrega un nodo a la red"""
        self.nodos[nodo.nombre] = nodo
    
    def obtener_nodo(self, nombre: str) -> NodoRedBayesiana:
        """Obtiene un nodo por nombre"""
        return self.nodos.get(nombre)
    
    def probabilidad_conjunta(self, asignacion: Dict[str, str]) -> float:
        """
        Calcula P(X1=x1, X2=x2, ..., Xn=xn)
        
        Usando la regla de la cadena:
        P(X1,...,Xn) = ∏_i P(Xi | Padres(Xi))
        """
        prob = 1.0
        
        for nombre_nodo, valor in asignacion.items():
            nodo = self.nodos[nombre_nodo]
            
            # Obtener valores de los padres
            valores_padres = tuple(asignacion[p.nombre] for p in nodo.padres)
            
            # Multiplicar por P(nodo | padres)
            prob *= nodo.obtener_probabilidad(valores_padres, valor)
        
        return prob
    
    def enumerar_asignaciones(self) -> List[Dict[str, str]]:
        """Genera todas las asignaciones posibles de variables"""
        nombres = list(self.nodos.keys())
        valores_por_nodo = [self.nodos[n].valores_posibles for n in nombres]
        
        asignaciones = []
        for combinacion in itertools.product(*valores_por_nodo):
            asignacion = dict(zip(nombres, combinacion))
            asignaciones.append(asignacion)
        
        return asignaciones


# Ejemplo 1: Red Bayesiana del Aspersor
def ejemplo_aspersor():
    """
    Ejemplo clásico: Red del Aspersor
    
    Nubosidad → Lluvia → Césped Mojado
                  ↓           ↑
              Aspersor -------┘
    """
    print("=== Red Bayesiana: Ejemplo del Aspersor ===\n")
    
    print("Estructura de la red:")
    print("  Nubosidad → Lluvia → Césped Mojado")
    print("                ↓           ↑")
    print("            Aspersor -------┘")
    print()
    
    # Crear nodos
    nubosidad = NodoRedBayesiana("nubosidad", ["si", "no"])
    lluvia = NodoRedBayesiana("lluvia", ["si", "no"])
    aspersor = NodoRedBayesiana("aspersor", ["si", "no"])
    cesped = NodoRedBayesiana("cesped_mojado", ["si", "no"])
    
    # Establecer dependencias
    lluvia.agregar_padre(nubosidad)
    aspersor.agregar_padre(nubosidad)
    cesped.agregar_padre(lluvia)
    cesped.agregar_padre(aspersor)
    
    # CPT de Nubosidad (sin padres)
    nubosidad.establecer_probabilidad((), "si", 0.5)
    nubosidad.establecer_probabilidad((), "no", 0.5)
    
    # CPT de Lluvia | Nubosidad
    lluvia.establecer_probabilidad(("si",), "si", 0.8)
    lluvia.establecer_probabilidad(("si",), "no", 0.2)
    lluvia.establecer_probabilidad(("no",), "si", 0.2)
    lluvia.establecer_probabilidad(("no",), "no", 0.8)
    
    # CPT de Aspersor | Nubosidad
    aspersor.establecer_probabilidad(("si",), "si", 0.1)
    aspersor.establecer_probabilidad(("si",), "no", 0.9)
    aspersor.establecer_probabilidad(("no",), "si", 0.5)
    aspersor.establecer_probabilidad(("no",), "no", 0.5)
    
    # CPT de Césped Mojado | Lluvia, Aspersor
    cesped.establecer_probabilidad(("si", "si"), "si", 0.99)
    cesped.establecer_probabilidad(("si", "si"), "no", 0.01)
    cesped.establecer_probabilidad(("si", "no"), "si", 0.90)
    cesped.establecer_probabilidad(("si", "no"), "no", 0.10)
    cesped.establecer_probabilidad(("no", "si"), "si", 0.90)
    cesped.establecer_probabilidad(("no", "si"), "no", 0.10)
    cesped.establecer_probabilidad(("no", "no"), "si", 0.01)
    cesped.establecer_probabilidad(("no", "no"), "no", 0.99)
    
    # Crear red
    red = RedBayesiana()
    red.agregar_nodo(nubosidad)
    red.agregar_nodo(lluvia)
    red.agregar_nodo(aspersor)
    red.agregar_nodo(cesped)
    
    # Calcular algunas probabilidades conjuntas
    print("Probabilidades Conjuntas (ejemplos):")
    
    asignaciones_ejemplo = [
        {"nubosidad": "si", "lluvia": "si", "aspersor": "no", "cesped_mojado": "si"},
        {"nubosidad": "no", "lluvia": "no", "aspersor": "si", "cesped_mojado": "si"},
        {"nubosidad": "no", "lluvia": "no", "aspersor": "no", "cesped_mojado": "no"},
    ]
    
    for asig in asignaciones_ejemplo:
        prob = red.probabilidad_conjunta(asig)
        print(f"  P({asig}) = {prob:.6f}")
    
    # Verificar normalización
    print("\nVerificación de normalización:")
    todas_asignaciones = red.enumerar_asignaciones()
    suma_total = sum(red.probabilidad_conjunta(asig) for asig in todas_asignaciones)
    print(f"  Suma de todas las probabilidades conjuntas = {suma_total:.6f}")
    print(f"  (Debe ser 1.0)")


# Ejemplo 2: Red Bayesiana de Diagnóstico Médico
def ejemplo_diagnostico():
    """Red bayesiana para diagnóstico médico simplificado"""
    print("\n\n=== Red Bayesiana: Diagnóstico Médico ===\n")
    
    print("Estructura:")
    print("  Gripe → Fiebre")
    print("  Gripe → Tos")
    print()
    
    # Crear nodos
    gripe = NodoRedBayesiana("gripe", ["si", "no"])
    fiebre = NodoRedBayesiana("fiebre", ["si", "no"])
    tos = NodoRedBayesiana("tos", ["si", "no"])
    
    # Dependencias
    fiebre.agregar_padre(gripe)
    tos.agregar_padre(gripe)
    
    # CPTs
    gripe.establecer_probabilidad((), "si", 0.1)
    gripe.establecer_probabilidad((), "no", 0.9)
    
    fiebre.establecer_probabilidad(("si",), "si", 0.9)
    fiebre.establecer_probabilidad(("si",), "no", 0.1)
    fiebre.establecer_probabilidad(("no",), "si", 0.1)
    fiebre.establecer_probabilidad(("no",), "no", 0.9)
    
    tos.establecer_probabilidad(("si",), "si", 0.8)
    tos.establecer_probabilidad(("si",), "no", 0.2)
    tos.establecer_probabilidad(("no",), "si", 0.2)
    tos.establecer_probabilidad(("no",), "no", 0.8)
    
    # Red
    red = RedBayesiana()
    red.agregar_nodo(gripe)
    red.agregar_nodo(fiebre)
    red.agregar_nodo(tos)
    
    print("Probabilidades a Priori:")
    print(f"  P(Gripe = si) = 0.1")
    print()
    
    print("Probabilidades Condicionales:")
    print("  P(Fiebre = si | Gripe = si) = 0.9")
    print("  P(Tos = si | Gripe = si) = 0.8")
    print()
    
    # Calcular P(Gripe, Fiebre=si, Tos=si)
    asig = {"gripe": "si", "fiebre": "si", "tos": "si"}
    prob = red.probabilidad_conjunta(asig)
    print(f"P(Gripe=si, Fiebre=si, Tos=si) = {prob:.4f}")
    
    asig2 = {"gripe": "no", "fiebre": "si", "tos": "si"}
    prob2 = red.probabilidad_conjunta(asig2)
    print(f"P(Gripe=no, Fiebre=si, Tos=si) = {prob2:.4f}")


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Redes Bayesianas ===\n")
    
    ejemplo_aspersor()
    ejemplo_diagnostico()
    
    print("\n" + "="*70)
    print("\nCaracterísticas de Redes Bayesianas:")
    print("  - Grafo acíclico dirigido (DAG)")
    print("  - Nodos = variables aleatorias")
    print("  - Aristas = dependencias probabilísticas")
    print("  - Cada nodo tiene CPT (Tabla de Probabilidad Condicional)")
    print("\nVentajas:")
    print("  - Representación compacta")
    print("  - Explota independencias condicionales")
    print("  - Permite inferencia eficiente")
    print("  - Interpretable (estructura causal)")
    print("\nRegla de la Cadena:")
    print("  P(X1,...,Xn) = ∏_i P(Xi | Padres(Xi))")

