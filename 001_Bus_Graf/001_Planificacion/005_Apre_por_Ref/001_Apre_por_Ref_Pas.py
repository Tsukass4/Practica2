"""
Algoritmo 33: Aprendizaje por Refuerzo Pasivo

En el aprendizaje por refuerzo pasivo, el agente aprende la función de valor
de una política fija observando las recompensas que recibe al seguirla.

Métodos:
- Estimación directa: Promediar retornos observados
- TD(0): Diferencia temporal con un paso
- TD(λ): Diferencia temporal con trazas de elegibilidad

No se modifica la política durante el aprendizaje.
"""

from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict


class EntornoGridWorld:
    """Mundo de cuadrícula simple para aprendizaje por refuerzo"""
    
    def __init__(self, filas: int = 3, columnas: int = 3):
        self.filas = filas
        self.columnas = columnas
        self.estado_actual = (0, 0)
        
        # Estados terminales
        self.terminales = {(2, 2): 10.0, (2, 0): -10.0}
        
        # Obstáculos
        self.obstaculos = {(1, 1)}
    
    def reiniciar(self) -> Tuple[int, int]:
        """Reinicia al estado inicial"""
        self.estado_actual = (0, 0)
        return self.estado_actual
    
    def es_terminal(self, estado: Tuple[int, int]) -> bool:
        """Verifica si un estado es terminal"""
        return estado in self.terminales
    
    def ejecutar_accion(self, accion: str) -> Tuple[Tuple[int, int], float, bool]:
        """
        Ejecuta una acción y retorna (nuevo_estado, recompensa, terminal)
        
        Acciones: 'arriba', 'abajo', 'izquierda', 'derecha'
        """
        if self.es_terminal(self.estado_actual):
            return self.estado_actual, 0.0, True
        
        fila, col = self.estado_actual
        
        # Determinar nuevo estado (con 80% de éxito, 20% de movimiento aleatorio)
        if random.random() < 0.8:
            accion_real = accion
        else:
            accion_real = random.choice(['arriba', 'abajo', 'izquierda', 'derecha'])
        
        if accion_real == 'arriba':
            nuevo_estado = (max(0, fila - 1), col)
        elif accion_real == 'abajo':
            nuevo_estado = (min(self.filas - 1, fila + 1), col)
        elif accion_real == 'izquierda':
            nuevo_estado = (fila, max(0, col - 1))
        else:  # derecha
            nuevo_estado = (fila, min(self.columnas - 1, col + 1))
        
        # No se puede entrar a obstáculos
        if nuevo_estado in self.obstaculos:
            nuevo_estado = self.estado_actual
        
        # Recompensa
        if nuevo_estado in self.terminales:
            recompensa = self.terminales[nuevo_estado]
        else:
            recompensa = -0.1  # Costo de vivir
        
        self.estado_actual = nuevo_estado
        terminal = self.es_terminal(nuevo_estado)
        
        return nuevo_estado, recompensa, terminal


class AprendizajePasivoDirecto:
    """Aprendizaje pasivo por estimación directa"""
    
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma
        self.retornos = defaultdict(list)  # Retornos observados por estado
        self.V = {}  # Función de valor estimada
    
    def aprender_episodio(self, episodio: List[Tuple[Tuple[int, int], float]]):
        """
        Aprende de un episodio completo.
        
        Args:
            episodio: Lista de (estado, recompensa)
        """
        # Calcular retornos para cada estado visitado
        G = 0
        for t in range(len(episodio) - 1, -1, -1):
            estado, recompensa = episodio[t]
            G = recompensa + self.gamma * G
            self.retornos[estado].append(G)
        
        # Actualizar función de valor (promedio de retornos)
        for estado in self.retornos:
            self.V[estado] = sum(self.retornos[estado]) / len(self.retornos[estado])
    
    def obtener_valor(self, estado: Tuple[int, int]) -> float:
        """Obtiene el valor estimado de un estado"""
        return self.V.get(estado, 0.0)


class AprendizajePasivoTD:
    """Aprendizaje pasivo por diferencia temporal TD(0)"""
    
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9):
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.V = defaultdict(float)  # Función de valor
    
    def actualizar(self, estado: Tuple[int, int], recompensa: float, 
                   estado_siguiente: Tuple[int, int], terminal: bool):
        """
        Actualización TD(0):
        V(s) ← V(s) + α[R + γV(s') - V(s)]
        """
        if terminal:
            objetivo = recompensa
        else:
            objetivo = recompensa + self.gamma * self.V[estado_siguiente]
        
        error_td = objetivo - self.V[estado]
        self.V[estado] += self.alpha * error_td
    
    def obtener_valor(self, estado: Tuple[int, int]) -> float:
        """Obtiene el valor estimado de un estado"""
        return self.V[estado]


# Ejemplo de uso
def ejemplo_aprendizaje_pasivo():
    """Compara estimación directa vs TD(0)"""
    print("=== Aprendizaje por Refuerzo Pasivo ===\n")
    
    # Crear entorno
    entorno = EntornoGridWorld()
    
    # Política fija (simple: ir hacia la derecha y arriba)
    def politica_fija(estado):
        fila, col = estado
        if col < 2:
            return 'derecha'
        else:
            return 'arriba'
    
    # Entrenar con estimación directa
    print("--- Estimación Directa ---")
    aprendiz_directo = AprendizajePasivoDirecto(gamma=0.9)
    
    for episodio_num in range(100):
        entorno.reiniciar()
        episodio = []
        
        for paso in range(50):
            estado = entorno.estado_actual
            accion = politica_fija(estado)
            nuevo_estado, recompensa, terminal = entorno.ejecutar_accion(accion)
            episodio.append((estado, recompensa))
            
            if terminal:
                break
        
        aprendiz_directo.aprender_episodio(episodio)
    
    print("Función de valor aprendida (Estimación Directa):")
    for fila in range(3):
        valores = []
        for col in range(3):
            estado = (fila, col)
            if estado in entorno.obstaculos:
                valores.append("  X  ")
            elif estado in entorno.terminales:
                valores.append(f"{entorno.terminales[estado]:5.1f}")
            else:
                valores.append(f"{aprendiz_directo.obtener_valor(estado):5.2f}")
        print("  ".join(valores))
    
    # Entrenar con TD(0)
    print("\n--- TD(0) ---")
    aprendiz_td = AprendizajePasivoTD(alpha=0.1, gamma=0.9)
    
    for episodio_num in range(100):
        entorno.reiniciar()
        
        for paso in range(50):
            estado = entorno.estado_actual
            accion = politica_fija(estado)
            nuevo_estado, recompensa, terminal = entorno.ejecutar_accion(accion)
            
            aprendiz_td.actualizar(estado, recompensa, nuevo_estado, terminal)
            
            if terminal:
                break
    
    print("Función de valor aprendida (TD(0)):")
    for fila in range(3):
        valores = []
        for col in range(3):
            estado = (fila, col)
            if estado in entorno.obstaculos:
                valores.append("  X  ")
            elif estado in entorno.terminales:
                valores.append(f"{entorno.terminales[estado]:5.1f}")
            else:
                valores.append(f"{aprendiz_td.obtener_valor(estado):5.2f}")
        print("  ".join(valores))
    
    print("\n" + "="*70)
    print("\nComparación:")
    print("\nEstimación Directa:")
    print("  + Simple de implementar")
    print("  + No sesgada")
    print("  - Requiere episodios completos")
    print("  - Convergencia lenta")
    print("\nTD(0):")
    print("  + Aprendizaje en línea (no requiere episodios completos)")
    print("  + Convergencia más rápida")
    print("  + Menor varianza")
    print("  - Sesgado inicialmente")


# Ejecutar ejemplo
if __name__ == "__main__":
    ejemplo_aprendizaje_pasivo()
    
    print("\n" + "="*70)
    print("\nCaracterísticas del Aprendizaje Pasivo:")
    print("- Aprende función de valor de una política fija")
    print("- No modifica la política")
    print("- Útil para evaluación de políticas")
    print("\nMétodos principales:")
    print("- Estimación Directa: Promedia retornos observados")
    print("- TD(0): V(s) ← V(s) + α[R + γV(s') - V(s)]")
    print("- TD(λ): Generalización con trazas de elegibilidad")

