"""
Algoritmo 41: Distribución de Probabilidad

Una distribución de probabilidad asigna probabilidades a todos los posibles
resultados de un experimento aleatorio.

Tipos:
- Distribución discreta: P(X = x) para valores discretos
- Distribución continua: f(x) función de densidad
- Distribución conjunta: P(X, Y)
- Distribución marginal: P(X) = Σ_y P(X, y)
"""

from typing import Dict, List, Tuple
import math
import random


class DistribucionDiscreta:
    """Distribución de probabilidad discreta"""
    
    def __init__(self, valores: List, probabilidades: List[float]):
        """
        Args:
            valores: Lista de valores posibles
            probabilidades: Probabilidades correspondientes
        """
        if len(valores) != len(probabilidades):
            raise ValueError("Valores y probabilidades deben tener la misma longitud")
        
        total = sum(probabilidades)
        if not math.isclose(total, 1.0, rel_tol=1e-5):
            raise ValueError(f"Las probabilidades deben sumar 1.0 (suma = {total})")
        
        self.distribucion = dict(zip(valores, probabilidades))
    
    def prob(self, valor) -> float:
        """Retorna P(X = valor)"""
        return self.distribucion.get(valor, 0.0)
    
    def esperanza(self) -> float:
        """Calcula E[X] = Σ x * P(x)"""
        return sum(x * p for x, p in self.distribucion.items())
    
    def varianza(self) -> float:
        """Calcula Var(X) = E[X²] - E[X]²"""
        media = self.esperanza()
        return sum((x - media)**2 * p for x, p in self.distribucion.items())
    
    def muestrear(self, n: int = 1) -> List:
        """Genera n muestras de la distribución"""
        valores = list(self.distribucion.keys())
        probs = list(self.distribucion.values())
        return random.choices(valores, weights=probs, k=n)


class DistribucionConjunta:
    """Distribución de probabilidad conjunta P(X, Y)"""
    
    def __init__(self):
        self.prob_conjunta = {}
    
    def establecer(self, x, y, probabilidad: float):
        """Establece P(X=x, Y=y)"""
        self.prob_conjunta[(x, y)] = probabilidad
    
    def prob(self, x, y) -> float:
        """Retorna P(X=x, Y=y)"""
        return self.prob_conjunta.get((x, y), 0.0)
    
    def marginal_x(self, x, valores_y: List) -> float:
        """Calcula P(X=x) = Σ_y P(X=x, Y=y)"""
        return sum(self.prob(x, y) for y in valores_y)
    
    def marginal_y(self, y, valores_x: List) -> float:
        """Calcula P(Y=y) = Σ_x P(X=x, Y=y)"""
        return sum(self.prob(x, y) for x in valores_x)
    
    def condicionada(self, x, dado_y, valores_x: List) -> float:
        """Calcula P(X=x | Y=y)"""
        prob_y = self.marginal_y(dado_y, valores_x)
        if prob_y == 0:
            return 0.0
        return self.prob(x, dado_y) / prob_y


# Distribuciones comunes

def distribucion_uniforme(valores: List):
    """Distribución uniforme discreta"""
    n = len(valores)
    probs = [1/n] * n
    return DistribucionDiscreta(valores, probs)


def distribucion_binomial(n: int, p: float):
    """Distribución binomial B(n, p)"""
    valores = list(range(n + 1))
    probabilidades = []
    
    for k in valores:
        # P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
        coef_binomial = math.comb(n, k)
        prob = coef_binomial * (p ** k) * ((1 - p) ** (n - k))
        probabilidades.append(prob)
    
    return DistribucionDiscreta(valores, probabilidades)


def distribucion_geometrica(p: float, max_val: int = 20):
    """Distribución geométrica (número de intentos hasta primer éxito)"""
    valores = list(range(1, max_val + 1))
    probabilidades = []
    
    for k in valores:
        # P(X = k) = (1-p)^(k-1) * p
        prob = ((1 - p) ** (k - 1)) * p
        probabilidades.append(prob)
    
    # Normalizar (truncada)
    total = sum(probabilidades)
    probabilidades = [p / total for p in probabilidades]
    
    return DistribucionDiscreta(valores, probabilidades)


# Ejemplos

def ejemplo_dado():
    """Ejemplo: Distribución de un dado"""
    print("=== Distribución de Probabilidad: Dado ===\n")
    
    # Dado justo
    dado_justo = distribucion_uniforme([1, 2, 3, 4, 5, 6])
    
    print("Dado Justo:")
    for valor in range(1, 7):
        print(f"  P(X = {valor}) = {dado_justo.prob(valor):.3f}")
    
    print(f"\n  E[X] = {dado_justo.esperanza():.2f}")
    print(f"  Var(X) = {dado_justo.varianza():.2f}")
    
    # Dado cargado
    dado_cargado = DistribucionDiscreta(
        [1, 2, 3, 4, 5, 6],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]  # Favorece el 6
    )
    
    print("\nDado Cargado (favorece 6):")
    for valor in range(1, 7):
        print(f"  P(X = {valor}) = {dado_cargado.prob(valor):.3f}")
    
    print(f"\n  E[X] = {dado_cargado.esperanza():.2f}")
    print(f"  Var(X) = {dado_cargado.varianza():.2f}")
    
    # Muestrear
    print("\n10 lanzamientos del dado cargado:")
    muestras = dado_cargado.muestrear(10)
    print(f"  {muestras}")


def ejemplo_moneda():
    """Ejemplo: Distribución binomial (lanzamientos de moneda)"""
    print("\n\n=== Distribución Binomial: Lanzamientos de Moneda ===\n")
    
    n = 10  # número de lanzamientos
    p = 0.5  # probabilidad de cara
    
    dist = distribucion_binomial(n, p)
    
    print(f"Lanzar una moneda {n} veces")
    print(f"P(Cara) = {p}")
    print("\nDistribución del número de caras:")
    
    for k in range(n + 1):
        prob = dist.prob(k)
        barra = '█' * int(prob * 100)
        print(f"  {k:2d} caras: {prob:.4f} {barra}")
    
    print(f"\nEsperanza (número promedio de caras): {dist.esperanza():.2f}")
    print(f"Varianza: {dist.varianza():.2f}")


def ejemplo_conjunta():
    """Ejemplo: Distribución conjunta"""
    print("\n\n=== Distribución Conjunta: Clima y Tráfico ===\n")
    
    dist = DistribucionConjunta()
    
    # P(Clima, Tráfico)
    print("Tabla de Probabilidad Conjunta:")
    print("                Tráfico Ligero  Tráfico Pesado")
    print("  Soleado           0.35            0.15")
    print("  Lluvioso          0.10            0.40")
    print()
    
    dist.establecer("soleado", "ligero", 0.35)
    dist.establecer("soleado", "pesado", 0.15)
    dist.establecer("lluvioso", "ligero", 0.10)
    dist.establecer("lluvioso", "pesado", 0.40)
    
    climas = ["soleado", "lluvioso"]
    traficos = ["ligero", "pesado"]
    
    # Marginales
    print("Distribuciones Marginales:")
    print("\nP(Clima):")
    for clima in climas:
        prob = dist.marginal_x(clima, traficos)
        print(f"  P({clima:10s}) = {prob:.2f}")
    
    print("\nP(Tráfico):")
    for trafico in traficos:
        prob = dist.marginal_y(trafico, climas)
        print(f"  P({trafico:10s}) = {prob:.2f}")
    
    # Condicionadas
    print("\nProbabilidades Condicionadas:")
    print("\nP(Tráfico | Clima = Lluvioso):")
    for trafico in traficos:
        prob = dist.condicionada(trafico, "lluvioso", traficos)
        print(f"  P({trafico:10s} | Lluvioso) = {prob:.2f}")


def ejemplo_geometrica():
    """Ejemplo: Distribución geométrica"""
    print("\n\n=== Distribución Geométrica: Intentos hasta Éxito ===\n")
    
    p = 0.2  # probabilidad de éxito
    dist = distribucion_geometrica(p, max_val=15)
    
    print(f"Probabilidad de éxito en cada intento: {p}")
    print("\nDistribución del número de intentos hasta el primer éxito:")
    
    for k in range(1, 11):
        prob = dist.prob(k)
        barra = '█' * int(prob * 100)
        print(f"  {k:2d} intentos: {prob:.4f} {barra}")
    
    print(f"\nEsperanza (intentos promedio): {dist.esperanza():.2f}")
    print(f"Teórico: 1/p = {1/p:.2f}")


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Distribuciones de Probabilidad ===\n")
    
    ejemplo_dado()
    ejemplo_moneda()
    ejemplo_conjunta()
    ejemplo_geometrica()
    
    print("\n" + "="*70)
    print("\nTipos de Distribuciones:")
    print("  - Uniforme: Todos los valores equiprobables")
    print("  - Binomial: Número de éxitos en n intentos")
    print("  - Geométrica: Intentos hasta primer éxito")
    print("  - Conjunta: Probabilidad de múltiples variables")
    print("\nPropiedades:")
    print("  - Σ P(x) = 1 (normalización)")
    print("  - E[X] = Σ x·P(x) (esperanza)")
    print("  - Var(X) = E[(X-μ)²] (varianza)")

