"""
Algoritmo 16: Búsqueda Online

La búsqueda online se realiza cuando el agente no conoce completamente el entorno
de antemano. Debe explorar mientras busca la solución, tomando decisiones basadas
en información local.

Algoritmos implementados:
- Online DFS (Depth-First Search)
- LRTA* (Learning Real-Time A*)
- Online A*

Características:
- El agente descubre el entorno mientras se mueve
- Debe tomar acciones sin conocer todo el espacio
- Aprende de la experiencia
- Útil en entornos dinámicos o desconocidos
"""

import random
import math
from typing import Dict, Tuple, List, Set, Optional


class EntornoLaberinto:
    """Simula un laberinto que el agente descubre gradualmente"""
    
    def __init__(self, ancho: int, alto: int, obstaculos: Set[Tuple[int, int]]):
        self.ancho = ancho
        self.alto = alto
        self.obstaculos = obstaculos
    
    def es_valido(self, pos: Tuple[int, int]) -> bool:
        """Verifica si una posición es válida"""
        x, y = pos
        return (0 <= x < self.ancho and 
                0 <= y < self.alto and 
                pos not in self.obstaculos)
    
    def obtener_acciones_validas(self, pos: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        """Retorna las acciones válidas desde una posición"""
        x, y = pos
        acciones = []
        
        movimientos = [
            ("arriba", (x, y - 1)),
            ("abajo", (x, y + 1)),
            ("izquierda", (x - 1, y)),
            ("derecha", (x + 1, y))
        ]
        
        for accion, nueva_pos in movimientos:
            if self.es_valido(nueva_pos):
                acciones.append((accion, nueva_pos))
        
        return acciones
    
    def costo_accion(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Retorna el costo de moverse entre dos posiciones"""
        return 1.0


def heuristica_manhattan(pos: Tuple[int, int], objetivo: Tuple[int, int]) -> float:
    """Heurística de distancia Manhattan"""
    return abs(pos[0] - objetivo[0]) + abs(pos[1] - objetivo[1])


class AgenteOnlineDFS:
    """Agente que realiza búsqueda online usando DFS"""
    
    def __init__(self, entorno: EntornoLaberinto, inicio: Tuple[int, int], objetivo: Tuple[int, int]):
        self.entorno = entorno
        self.inicio = inicio
        self.objetivo = objetivo
        self.visitados: Set[Tuple[int, int]] = set()
        self.pila_retroceso: List[Tuple[int, int]] = []
        self.camino: List[Tuple[int, int]] = [inicio]
    
    def buscar(self, max_pasos: int = 1000) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Realiza búsqueda online DFS.
        
        Returns:
            Tupla con (exito, camino, pasos)
        """
        pos_actual = self.inicio
        pasos = 0
        
        while pasos < max_pasos:
            pasos += 1
            self.visitados.add(pos_actual)
            
            # Verificar si alcanzamos el objetivo
            if pos_actual == self.objetivo:
                return True, self.camino, pasos
            
            # Obtener acciones válidas no visitadas
            acciones = self.entorno.obtener_acciones_validas(pos_actual)
            acciones_no_visitadas = [(a, p) for a, p in acciones if p not in self.visitados]
            
            if acciones_no_visitadas:
                # Elegir una acción no visitada (primera disponible)
                accion, nueva_pos = acciones_no_visitadas[0]
                self.pila_retroceso.append(pos_actual)
                pos_actual = nueva_pos
                self.camino.append(nueva_pos)
            elif self.pila_retroceso:
                # Retroceder
                pos_actual = self.pila_retroceso.pop()
                self.camino.append(pos_actual)
            else:
                # No hay más opciones
                return False, self.camino, pasos
        
        return False, self.camino, pasos


class AgenteLRTAEstrella:
    """Agente que realiza LRTA* (Learning Real-Time A*)"""
    
    def __init__(self, entorno: EntornoLaberinto, inicio: Tuple[int, int], objetivo: Tuple[int, int]):
        self.entorno = entorno
        self.inicio = inicio
        self.objetivo = objetivo
        # Tabla H: estimaciones heurísticas aprendidas
        self.H: Dict[Tuple[int, int], float] = {}
        self.camino: List[Tuple[int, int]] = [inicio]
    
    def obtener_h(self, pos: Tuple[int, int]) -> float:
        """Obtiene el valor heurístico (aprendido o inicial)"""
        if pos not in self.H:
            self.H[pos] = heuristica_manhattan(pos, self.objetivo)
        return self.H[pos]
    
    def buscar(self, max_pasos: int = 1000) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Realiza búsqueda LRTA*.
        
        Returns:
            Tupla con (exito, camino, pasos)
        """
        pos_actual = self.inicio
        pasos = 0
        
        while pasos < max_pasos:
            pasos += 1
            
            # Verificar si alcanzamos el objetivo
            if pos_actual == self.objetivo:
                return True, self.camino, pasos
            
            # Obtener acciones válidas
            acciones = self.entorno.obtener_acciones_validas(pos_actual)
            
            if not acciones:
                return False, self.camino, pasos
            
            # Calcular f(s') = c(s,a,s') + H(s') para cada sucesor
            mejor_accion = None
            mejor_pos = None
            mejor_f = float('inf')
            segundo_mejor_f = float('inf')
            
            for accion, nueva_pos in acciones:
                costo = self.entorno.costo_accion(pos_actual, nueva_pos)
                f_valor = costo + self.obtener_h(nueva_pos)
                
                if f_valor < mejor_f:
                    segundo_mejor_f = mejor_f
                    mejor_f = f_valor
                    mejor_accion = accion
                    mejor_pos = nueva_pos
                elif f_valor < segundo_mejor_f:
                    segundo_mejor_f = f_valor
            
            # Actualizar H(s) con el segundo mejor valor (aprendizaje)
            # H(s) = min(H(s), segundo_mejor_f)
            if len(acciones) > 1:
                self.H[pos_actual] = min(self.obtener_h(pos_actual), segundo_mejor_f)
            else:
                self.H[pos_actual] = mejor_f
            
            # Moverse al mejor sucesor
            pos_actual = mejor_pos
            self.camino.append(mejor_pos)
        
        return False, self.camino, pasos


class AgenteOnlineAEstrella:
    """Agente que realiza búsqueda online A*"""
    
    def __init__(self, entorno: EntornoLaberinto, inicio: Tuple[int, int], objetivo: Tuple[int, int]):
        self.entorno = entorno
        self.inicio = inicio
        self.objetivo = objetivo
        self.g: Dict[Tuple[int, int], float] = {inicio: 0}
        self.camino: List[Tuple[int, int]] = [inicio]
        self.explorados: Set[Tuple[int, int]] = set()
    
    def buscar(self, max_pasos: int = 1000, horizonte: int = 5) -> Tuple[bool, List[Tuple[int, int]], int]:
        """
        Realiza búsqueda online A* con horizonte limitado.
        
        Args:
            max_pasos: Máximo número de pasos
            horizonte: Profundidad de planificación local
        
        Returns:
            Tupla con (exito, camino, pasos)
        """
        pos_actual = self.inicio
        pasos = 0
        
        while pasos < max_pasos:
            pasos += 1
            
            # Verificar si alcanzamos el objetivo
            if pos_actual == self.objetivo:
                return True, self.camino, pasos
            
            # Planificar localmente usando A* con horizonte limitado
            mejor_accion = self._planificar_local(pos_actual, horizonte)
            
            if mejor_accion is None:
                return False, self.camino, pasos
            
            # Ejecutar la mejor acción
            accion, nueva_pos = mejor_accion
            self.g[nueva_pos] = self.g.get(pos_actual, 0) + 1
            pos_actual = nueva_pos
            self.camino.append(nueva_pos)
        
        return False, self.camino, pasos
    
    def _planificar_local(self, pos: Tuple[int, int], horizonte: int) -> Optional[Tuple[str, Tuple[int, int]]]:
        """Planifica localmente usando A* con horizonte limitado"""
        acciones = self.entorno.obtener_acciones_validas(pos)
        
        if not acciones:
            return None
        
        # Evaluar cada acción
        mejor_accion = None
        mejor_valor = float('inf')
        
        for accion, nueva_pos in acciones:
            # Valor = g + h
            g_nuevo = self.g.get(pos, 0) + 1
            h_nuevo = heuristica_manhattan(nueva_pos, self.objetivo)
            f_nuevo = g_nuevo + h_nuevo
            
            if f_nuevo < mejor_valor:
                mejor_valor = f_nuevo
                mejor_accion = (accion, nueva_pos)
        
        return mejor_accion


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Búsqueda Online ===\n")
    
    # Crear laberinto
    obstaculos = {
        (2, 1), (2, 2), (2, 3),
        (5, 2), (5, 3), (5, 4),
        (7, 1), (7, 2)
    }
    entorno = EntornoLaberinto(10, 8, obstaculos)
    
    inicio = (0, 0)
    objetivo = (9, 7)
    
    print(f"Inicio: {inicio}")
    print(f"Objetivo: {objetivo}\n")
    
    # Online DFS
    print("--- Online DFS ---")
    agente_dfs = AgenteOnlineDFS(entorno, inicio, objetivo)
    exito1, camino1, pasos1 = agente_dfs.buscar(max_pasos=500)
    print(f"Éxito: {'✓ Sí' if exito1 else '✗ No'}")
    print(f"Longitud del camino: {len(camino1)}")
    print(f"Pasos: {pasos1}\n")
    
    # LRTA*
    print("--- LRTA* (Learning Real-Time A*) ---")
    agente_lrta = AgenteLRTAEstrella(entorno, inicio, objetivo)
    exito2, camino2, pasos2 = agente_lrta.buscar(max_pasos=500)
    print(f"Éxito: {'✓ Sí' if exito2 else '✗ No'}")
    print(f"Longitud del camino: {len(camino2)}")
    print(f"Pasos: {pasos2}")
    print(f"Valores aprendidos (muestra): {list(agente_lrta.H.items())[:5]}\n")
    
    # Online A*
    print("--- Online A* ---")
    agente_astar = AgenteOnlineAEstrella(entorno, inicio, objetivo)
    exito3, camino3, pasos3 = agente_astar.buscar(max_pasos=500, horizonte=5)
    print(f"Éxito: {'✓ Sí' if exito3 else '✗ No'}")
    print(f"Longitud del camino: {len(camino3)}")
    print(f"Pasos: {pasos3}\n")
    
    print("="*50)
    print("\nCaracterísticas de la Búsqueda Online:")
    print("- El agente descubre el entorno mientras se mueve")
    print("- Decisiones basadas en información local")
    print("- LRTA* aprende valores heurísticos mejorados")
    print("- Útil cuando el entorno es desconocido o dinámico")
    print("\nComparación:")
    print(f"DFS:     {pasos1} pasos, camino de {len(camino1)}")
    print(f"LRTA*:   {pasos2} pasos, camino de {len(camino2)}")
    print(f"A*:      {pasos3} pasos, camino de {len(camino3)}")

