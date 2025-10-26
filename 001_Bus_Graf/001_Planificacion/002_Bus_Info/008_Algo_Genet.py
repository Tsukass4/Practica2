"""
Algoritmo 15: Algoritmos Genéticos (Genetic Algorithms)

Los algoritmos genéticos son metaheurísticas inspiradas en la evolución biológica.
Usan selección, cruce y mutación para evolucionar una población de soluciones.

Características:
- Población de soluciones (no una sola)
- Selección basada en fitness
- Cruce (recombinación) de soluciones
- Mutación para diversidad
- Evolución iterativa
"""

import random
from typing import List, Tuple, Callable


class IndividuoReinas:
    """Representa un individuo (solución) para el problema de las N-Reinas"""
    
    def __init__(self, genes: List[int], n: int = 8):
        self.genes = genes  # Posición de cada reina
        self.n = n
        self.fitness = self._calcular_fitness()
    
    def _calcular_fitness(self) -> float:
        """
        Calcula el fitness (aptitud) del individuo.
        Mayor fitness = mejor solución.
        """
        ataques = 0
        for i in range(len(self.genes)):
            for j in range(i + 1, len(self.genes)):
                if self.genes[i] == self.genes[j]:
                    ataques += 1
                elif abs(self.genes[i] - self.genes[j]) == abs(i - j):
                    ataques += 1
        
        # Fitness = pares que NO se atacan
        max_pares = (self.n * (self.n - 1)) // 2
        return max_pares - ataques
    
    def es_solucion(self) -> bool:
        """Verifica si es una solución válida"""
        max_pares = (self.n * (self.n - 1)) // 2
        return self.fitness == max_pares
    
    def __repr__(self):
        return f"Individuo(genes={self.genes}, fitness={self.fitness})"


def crear_poblacion_inicial(tamano: int, n: int) -> List[IndividuoReinas]:
    """Crea una población inicial aleatoria"""
    poblacion = []
    for _ in range(tamano):
        genes = [random.randint(0, n - 1) for _ in range(n)]
        poblacion.append(IndividuoReinas(genes, n))
    return poblacion


def seleccion_torneo(poblacion: List[IndividuoReinas], k: int = 3) -> IndividuoReinas:
    """
    Selección por torneo: elige k individuos aleatorios y retorna el mejor.
    """
    competidores = random.sample(poblacion, k)
    return max(competidores, key=lambda ind: ind.fitness)


def seleccion_ruleta(poblacion: List[IndividuoReinas]) -> IndividuoReinas:
    """
    Selección por ruleta: probabilidad proporcional al fitness.
    """
    fitness_total = sum(ind.fitness for ind in poblacion)
    if fitness_total == 0:
        return random.choice(poblacion)
    
    probabilidades = [ind.fitness / fitness_total for ind in poblacion]
    return random.choices(poblacion, weights=probabilidades, k=1)[0]


def cruce_un_punto(padre1: IndividuoReinas, padre2: IndividuoReinas) -> Tuple[IndividuoReinas, IndividuoReinas]:
    """
    Cruce de un punto: divide en un punto y combina.
    """
    n = padre1.n
    punto = random.randint(1, n - 1)
    
    hijo1_genes = padre1.genes[:punto] + padre2.genes[punto:]
    hijo2_genes = padre2.genes[:punto] + padre1.genes[punto:]
    
    return IndividuoReinas(hijo1_genes, n), IndividuoReinas(hijo2_genes, n)


def cruce_dos_puntos(padre1: IndividuoReinas, padre2: IndividuoReinas) -> Tuple[IndividuoReinas, IndividuoReinas]:
    """
    Cruce de dos puntos: intercambia segmento entre dos puntos.
    """
    n = padre1.n
    punto1 = random.randint(1, n - 2)
    punto2 = random.randint(punto1 + 1, n - 1)
    
    hijo1_genes = padre1.genes[:punto1] + padre2.genes[punto1:punto2] + padre1.genes[punto2:]
    hijo2_genes = padre2.genes[:punto1] + padre1.genes[punto1:punto2] + padre2.genes[punto2:]
    
    return IndividuoReinas(hijo1_genes, n), IndividuoReinas(hijo2_genes, n)


def cruce_uniforme(padre1: IndividuoReinas, padre2: IndividuoReinas) -> Tuple[IndividuoReinas, IndividuoReinas]:
    """
    Cruce uniforme: cada gen se hereda aleatoriamente de uno de los padres.
    """
    n = padre1.n
    hijo1_genes = []
    hijo2_genes = []
    
    for i in range(n):
        if random.random() < 0.5:
            hijo1_genes.append(padre1.genes[i])
            hijo2_genes.append(padre2.genes[i])
        else:
            hijo1_genes.append(padre2.genes[i])
            hijo2_genes.append(padre1.genes[i])
    
    return IndividuoReinas(hijo1_genes, n), IndividuoReinas(hijo2_genes, n)


def mutacion(individuo: IndividuoReinas, tasa_mutacion: float = 0.1) -> IndividuoReinas:
    """
    Mutación: cambia aleatoriamente algunos genes.
    """
    genes_mutados = individuo.genes.copy()
    
    for i in range(len(genes_mutados)):
        if random.random() < tasa_mutacion:
            genes_mutados[i] = random.randint(0, individuo.n - 1)
    
    return IndividuoReinas(genes_mutados, individuo.n)


def algoritmo_genetico(n: int = 8,
                      tamano_poblacion: int = 100,
                      generaciones: int = 1000,
                      tasa_cruce: float = 0.8,
                      tasa_mutacion: float = 0.1,
                      elitismo: int = 2) -> Tuple[IndividuoReinas, int, List[float]]:
    """
    Algoritmo genético para el problema de las N-Reinas.
    
    Args:
        n: Tamaño del tablero
        tamano_poblacion: Tamaño de la población
        generaciones: Número máximo de generaciones
        tasa_cruce: Probabilidad de cruce
        tasa_mutacion: Probabilidad de mutación por gen
        elitismo: Número de mejores individuos a preservar
    
    Returns:
        Tupla con (mejor_individuo, generacion_encontrada, historial_fitness)
    """
    # Crear población inicial
    poblacion = crear_poblacion_inicial(tamano_poblacion, n)
    
    mejor_individuo = max(poblacion, key=lambda ind: ind.fitness)
    historial_fitness = [mejor_individuo.fitness]
    
    for generacion in range(generaciones):
        # Ordenar población por fitness
        poblacion.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Actualizar mejor individuo
        if poblacion[0].fitness > mejor_individuo.fitness:
            mejor_individuo = poblacion[0]
        
        historial_fitness.append(mejor_individuo.fitness)
        
        # Si encontramos solución, terminar
        if mejor_individuo.es_solucion():
            return mejor_individuo, generacion, historial_fitness
        
        # Nueva población
        nueva_poblacion = []
        
        # Elitismo: preservar los mejores
        nueva_poblacion.extend(poblacion[:elitismo])
        
        # Generar resto de la población
        while len(nueva_poblacion) < tamano_poblacion:
            # Selección
            padre1 = seleccion_torneo(poblacion)
            padre2 = seleccion_torneo(poblacion)
            
            # Cruce
            if random.random() < tasa_cruce:
                hijo1, hijo2 = cruce_un_punto(padre1, padre2)
            else:
                hijo1, hijo2 = padre1, padre2
            
            # Mutación
            hijo1 = mutacion(hijo1, tasa_mutacion)
            hijo2 = mutacion(hijo2, tasa_mutacion)
            
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < tamano_poblacion:
                nueva_poblacion.append(hijo2)
        
        poblacion = nueva_poblacion
    
    return mejor_individuo, generaciones, historial_fitness


# Ejemplo de uso
if __name__ == "__main__":
    print("=== Algoritmos Genéticos ===\n")
    print("Problema: 8-Reinas\n")
    
    # Ejecutar algoritmo genético
    print("--- Ejecución del Algoritmo Genético ---")
    mejor, gen, historial = algoritmo_genetico(
        n=8,
        tamano_poblacion=100,
        generaciones=1000,
        tasa_cruce=0.8,
        tasa_mutacion=0.1,
        elitismo=2
    )
    
    max_fitness = (8 * 7) // 2
    print(f"Mejor individuo: {mejor.genes}")
    print(f"Fitness: {mejor.fitness}/{max_fitness}")
    print(f"Generación: {gen}")
    print(f"¿Solución?: {'✓ Sí' if mejor.es_solucion() else '✗ No'}\n")
    
    # Múltiples ejecuciones
    print("--- Comparación (10 ejecuciones) ---")
    exitos = 0
    total_generaciones = 0
    
    for i in range(10):
        mejor_i, gen_i, _ = algoritmo_genetico(
            n=8,
            tamano_poblacion=100,
            generaciones=500,
            tasa_cruce=0.8,
            tasa_mutacion=0.1
        )
        total_generaciones += gen_i
        if mejor_i.es_solucion():
            exitos += 1
    
    print(f"Soluciones encontradas: {exitos}/10")
    print(f"Generaciones promedio: {total_generaciones/10:.1f}\n")
    
    # Probar diferentes parámetros
    print("--- Comparación de Parámetros ---")
    configs = [
        (50, 0.8, 0.05, "Población pequeña, mutación baja"),
        (100, 0.8, 0.1, "Configuración estándar"),
        (200, 0.9, 0.15, "Población grande, mutación alta"),
    ]
    
    for tam_pob, t_cruce, t_mut, desc in configs:
        mejor_c, gen_c, _ = algoritmo_genetico(
            n=8,
            tamano_poblacion=tam_pob,
            generaciones=500,
            tasa_cruce=t_cruce,
            tasa_mutacion=t_mut
        )
        print(f"{desc}:")
        print(f"  Fitness: {mejor_c.fitness}/{max_fitness}, Generación: {gen_c}")
    
    print("\n" + "="*50)
    print("\nCaracterísticas de los Algoritmos Genéticos:")
    print("- Inspirados en la evolución biológica")
    print("- Población de soluciones que evoluciona")
    print("- Operadores: Selección, Cruce, Mutación")
    print("- Elitismo preserva las mejores soluciones")
    print("- Balance exploración-explotación")
    print("\nVentajas:")
    print("- Robusto para problemas complejos")
    print("- Paralelizable naturalmente")
    print("- No requiere información del gradiente")

