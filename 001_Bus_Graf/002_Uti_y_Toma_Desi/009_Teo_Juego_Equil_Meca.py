"""
Algoritmo 32: Teoría de Juegos - Equilibrios y Mecanismos

La teoría de juegos estudia la toma de decisiones estratégicas cuando
múltiples agentes interactúan. Los conceptos clave incluyen:

- Equilibrio de Nash: Ningún jugador puede mejorar cambiando unilateralmente
- Estrategias dominantes
- Juegos de suma cero
- Juegos cooperativos vs no cooperativos
- Diseño de mecanismos

Este archivo implementa conceptos fundamentales de teoría de juegos.
"""

from typing import Dict, List, Tuple, Optional, Set
import itertools


class JuegoNormal:
    """Juego en forma normal (matriz de pagos)"""
    
    def __init__(self, jugadores: List[str], 
                 estrategias: Dict[str, List[str]],
                 utilidades: Dict[Tuple, Dict[str, float]]):
        """
        Args:
            jugadores: Lista de jugadores
            estrategias: Estrategias disponibles para cada jugador
            utilidades: Utilidad para cada combinación de estrategias
                       {(estrategia_j1, estrategia_j2, ...): {jugador: utilidad}}
        """
        self.jugadores = jugadores
        self.estrategias = estrategias
        self.utilidades = utilidades
    
    def obtener_utilidad(self, perfil_estrategias: Tuple[str, ...], jugador: str) -> float:
        """Obtiene la utilidad de un jugador dado un perfil de estrategias"""
        return self.utilidades.get(perfil_estrategias, {}).get(jugador, 0.0)
    
    def mejor_respuesta(self, jugador: str, estrategias_otros: Tuple[str, ...]) -> List[str]:
        """
        Encuentra la(s) mejor(es) respuesta(s) de un jugador dadas las estrategias de los demás.
        
        Args:
            jugador: Jugador que responde
            estrategias_otros: Estrategias de los otros jugadores
        
        Returns:
            Lista de mejores respuestas
        """
        idx_jugador = self.jugadores.index(jugador)
        mejor_utilidad = float('-inf')
        mejores_respuestas = []
        
        for estrategia in self.estrategias[jugador]:
            # Construir perfil completo
            perfil = list(estrategias_otros)
            perfil.insert(idx_jugador, estrategia)
            perfil_tuple = tuple(perfil)
            
            utilidad = self.obtener_utilidad(perfil_tuple, jugador)
            
            if utilidad > mejor_utilidad:
                mejor_utilidad = utilidad
                mejores_respuestas = [estrategia]
            elif utilidad == mejor_utilidad:
                mejores_respuestas.append(estrategia)
        
        return mejores_respuestas
    
    def es_equilibrio_nash(self, perfil_estrategias: Tuple[str, ...]) -> bool:
        """
        Verifica si un perfil de estrategias es un equilibrio de Nash.
        
        Un perfil es equilibrio de Nash si ningún jugador puede mejorar
        cambiando unilateralmente su estrategia.
        """
        for i, jugador in enumerate(self.jugadores):
            # Estrategias de los otros jugadores
            estrategias_otros = tuple(s for j, s in enumerate(perfil_estrategias) if j != i)
            
            # Mejor respuesta del jugador
            mejores_respuestas = self.mejor_respuesta(jugador, estrategias_otros)
            
            # Si la estrategia actual no es mejor respuesta, no es equilibrio
            if perfil_estrategias[i] not in mejores_respuestas:
                return False
        
        return True
    
    def encontrar_equilibrios_nash(self) -> List[Tuple[str, ...]]:
        """Encuentra todos los equilibrios de Nash en estrategias puras"""
        equilibrios = []
        
        # Generar todos los perfiles posibles
        estrategias_por_jugador = [self.estrategias[j] for j in self.jugadores]
        
        for perfil in itertools.product(*estrategias_por_jugador):
            if self.es_equilibrio_nash(perfil):
                equilibrios.append(perfil)
        
        return equilibrios
    
    def estrategia_dominante(self, jugador: str) -> Optional[str]:
        """
        Encuentra la estrategia estrictamente dominante de un jugador, si existe.
        
        Una estrategia s domina a s' si u(s, s_{-i}) > u(s', s_{-i}) para
        todas las estrategias de los demás jugadores.
        """
        estrategias_candidatas = self.estrategias[jugador].copy()
        
        for candidata in self.estrategias[jugador]:
            es_dominante = True
            
            # Generar todas las combinaciones de estrategias de los demás
            otros_jugadores = [j for j in self.jugadores if j != jugador]
            estrategias_otros = [self.estrategias[j] for j in otros_jugadores]
            
            for perfil_otros in itertools.product(*estrategias_otros):
                # Construir perfiles completos
                idx = self.jugadores.index(jugador)
                
                for otra_estrategia in self.estrategias[jugador]:
                    if otra_estrategia == candidata:
                        continue
                    
                    perfil_candidata = list(perfil_otros)
                    perfil_candidata.insert(idx, candidata)
                    
                    perfil_otra = list(perfil_otros)
                    perfil_otra.insert(idx, otra_estrategia)
                    
                    util_candidata = self.obtener_utilidad(tuple(perfil_candidata), jugador)
                    util_otra = self.obtener_utilidad(tuple(perfil_otra), jugador)
                    
                    if util_candidata <= util_otra:
                        es_dominante = False
                        break
                
                if not es_dominante:
                    break
            
            if es_dominante:
                return candidata
        
        return None


# Ejemplo 1: Dilema del Prisionero
def ejemplo_dilema_prisionero():
    """
    Dilema del Prisionero clásico:
    
    Dos prisioneros pueden cooperar (C) o delatar (D).
    
    Matriz de pagos (años en prisión, negativo):
                Prisionero 2
                C       D
    Pris. 1 C  -1,-1   -3,0
            D   0,-3   -2,-2
    """
    print("=== Ejemplo: Dilema del Prisionero ===\n")
    
    jugadores = ["P1", "P2"]
    estrategias = {
        "P1": ["C", "D"],
        "P2": ["C", "D"]
    }
    
    utilidades = {
        ("C", "C"): {"P1": -1, "P2": -1},
        ("C", "D"): {"P1": -3, "P2": 0},
        ("D", "C"): {"P1": 0, "P2": -3},
        ("D", "D"): {"P1": -2, "P2": -2}
    }
    
    juego = JuegoNormal(jugadores, estrategias, utilidades)
    
    # Mostrar matriz de pagos
    print("Matriz de pagos (años en prisión):")
    print("              P2: C      P2: D")
    print(f"P1: C      {utilidades[('C','C')]['P1']},{utilidades[('C','C')]['P2']}      {utilidades[('C','D')]['P1']},{utilidades[('C','D')]['P2']}")
    print(f"P1: D      {utilidades[('D','C')]['P1']},{utilidades[('D','C')]['P2']}      {utilidades[('D','D')]['P1']},{utilidades[('D','D')]['P2']}")
    print()
    
    # Encontrar estrategias dominantes
    for jugador in jugadores:
        dom = juego.estrategia_dominante(jugador)
        if dom:
            print(f"Estrategia dominante de {jugador}: {dom}")
    
    # Encontrar equilibrios de Nash
    equilibrios = juego.encontrar_equilibrios_nash()
    print(f"\nEquilibrios de Nash: {equilibrios}")
    
    # Análisis
    print("\nAnálisis:")
    print("- (D, D) es el único equilibrio de Nash")
    print("- Pero (C, C) daría mejor resultado para ambos")
    print("- Este es el dilema: la racionalidad individual lleva a resultado subóptimo")


# Ejemplo 2: Batalla de los Sexos
def ejemplo_batalla_sexos():
    """
    Batalla de los Sexos:
    
    Pareja decide entre Ópera (O) o Fútbol (F).
    Prefieren estar juntos, pero tienen preferencias diferentes.
    
                Ella
                O   F
    Él      O  2,1  0,0
            F  0,0  1,2
    """
    print("\n\n=== Ejemplo: Batalla de los Sexos ===\n")
    
    jugadores = ["El", "Ella"]
    estrategias = {
        "El": ["O", "F"],
        "Ella": ["O", "F"]
    }
    
    utilidades = {
        ("O", "O"): {"El": 2, "Ella": 1},
        ("O", "F"): {"El": 0, "Ella": 0},
        ("F", "O"): {"El": 0, "Ella": 0},
        ("F", "F"): {"El": 1, "Ella": 2}
    }
    
    juego = JuegoNormal(jugadores, estrategias, utilidades)
    
    print("Matriz de pagos:")
    print("          Ella: O  Ella: F")
    print(f"Él: O      2,1      0,0")
    print(f"Él: F      0,0      1,2")
    print()
    
    equilibrios = juego.encontrar_equilibrios_nash()
    print(f"Equilibrios de Nash: {equilibrios}")
    
    print("\nAnálisis:")
    print("- Dos equilibrios de Nash: (O, O) y (F, F)")
    print("- Ambos prefieren coordinar que no coordinar")
    print("- Problema de coordinación: ¿cuál equilibrio elegir?")


# Ejemplo 3: Piedra, Papel o Tijera
def ejemplo_piedra_papel_tijera():
    """
    Piedra, Papel o Tijera:
    Juego de suma cero sin equilibrio en estrategias puras.
    """
    print("\n\n=== Ejemplo: Piedra, Papel o Tijera ===\n")
    
    jugadores = ["J1", "J2"]
    estrategias = {
        "J1": ["Piedra", "Papel", "Tijera"],
        "J2": ["Piedra", "Papel", "Tijera"]
    }
    
    utilidades = {
        ("Piedra", "Piedra"): {"J1": 0, "J2": 0},
        ("Piedra", "Papel"): {"J1": -1, "J2": 1},
        ("Piedra", "Tijera"): {"J1": 1, "J2": -1},
        ("Papel", "Piedra"): {"J1": 1, "J2": -1},
        ("Papel", "Papel"): {"J1": 0, "J2": 0},
        ("Papel", "Tijera"): {"J1": -1, "J2": 1},
        ("Tijera", "Piedra"): {"J1": -1, "J2": 1},
        ("Tijera", "Papel"): {"J1": 1, "J2": -1},
        ("Tijera", "Tijera"): {"J1": 0, "J2": 0}
    }
    
    juego = JuegoNormal(jugadores, estrategias, utilidades)
    
    equilibrios = juego.encontrar_equilibrios_nash()
    print(f"Equilibrios de Nash en estrategias puras: {equilibrios if equilibrios else 'Ninguno'}")
    
    print("\nAnálisis:")
    print("- Juego de suma cero (las ganancias suman 0)")
    print("- No tiene equilibrio en estrategias puras")
    print("- El equilibrio de Nash está en estrategias mixtas:")
    print("  Cada jugador elige cada opción con probabilidad 1/3")


# Ejemplo 4: Subasta de segundo precio (Mecanismo de Vickrey)
def ejemplo_subasta_vickrey():
    """
    Subasta de segundo precio (Vickrey):
    - Cada participante hace una oferta sellada
    - Gana el que ofrece más
    - Pero paga el segundo precio más alto
    
    Propiedad: Decir la verdad es estrategia dominante
    """
    print("\n\n=== Ejemplo: Subasta de Vickrey (Segundo Precio) ===\n")
    
    print("Mecanismo:")
    print("1. Cada participante envía oferta sellada")
    print("2. Gana quien ofrece más")
    print("3. Paga el segundo precio más alto")
    print()
    
    # Valores verdaderos
    valores_verdaderos = {"A": 100, "B": 80, "C": 60}
    
    # Ofertas (estrategia: decir la verdad)
    ofertas_verdad = {"A": 100, "B": 80, "C": 60}
    
    # Determinar ganador y precio
    ganador = max(ofertas_verdad, key=ofertas_verdad.get)
    ofertas_ordenadas = sorted(ofertas_verdad.values(), reverse=True)
    precio = ofertas_ordenadas[1]  # Segundo precio
    
    print(f"Valores verdaderos: {valores_verdaderos}")
    print(f"Ofertas (verdad): {ofertas_verdad}")
    print(f"\nGanador: {ganador}")
    print(f"Precio pagado: ${precio}")
    print(f"Utilidad del ganador: ${valores_verdaderos[ganador] - precio}")
    
    # Demostrar que mentir no ayuda
    print("\n--- ¿Qué pasa si A miente? ---")
    ofertas_mentira = {"A": 70, "B": 80, "C": 60}  # A ofrece menos
    
    ganador_mentira = max(ofertas_mentira, key=ofertas_mentira.get)
    ofertas_ord_mentira = sorted(ofertas_mentira.values(), reverse=True)
    precio_mentira = ofertas_ord_mentira[1]
    
    print(f"Ofertas (A miente): {ofertas_mentira}")
    print(f"Ganador: {ganador_mentira}")
    print(f"Resultado para A: Pierde la subasta (utilidad = 0)")
    print(f"\nConclusión: Decir la verdad es estrategia dominante")


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Teoría de Juegos: Equilibrios y Mecanismos ===\n")
    
    ejemplo_dilema_prisionero()
    ejemplo_batalla_sexos()
    ejemplo_piedra_papel_tijera()
    ejemplo_subasta_vickrey()
    
    print("\n" + "="*70)
    print("\nConceptos clave:")
    print("- Equilibrio de Nash: Ningún jugador mejora cambiando unilateralmente")
    print("- Estrategia dominante: Mejor respuesta independiente de otros")
    print("- Juego de suma cero: Las ganancias suman cero")
    print("- Diseño de mecanismos: Diseñar reglas para obtener resultados deseados")
    print("\nAplicaciones:")
    print("- Economía: Subastas, mercados, competencia")
    print("- IA: Agentes múltiples, negociación")
    print("- Política: Votación, tratados")
    print("- Biología: Evolución, comportamiento animal")

