"""
Algoritmo 61: Naïve Bayes
Clasificador probabilístico basado en independencia condicional.
P(C|X1,...,Xn) ∝ P(C) ∏ P(Xi|C)
"""
from collections import defaultdict
import math

class NaiveBayes:
    def __init__(self):
        self.clases = set()
        self.prior = {}
        self.likelihood = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    def entrenar(self, X, y):
        # Contar frecuencias
        n = len(y)
        conteos_clase = defaultdict(int)
        conteos_feature = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for features, clase in zip(X, y):
            self.clases.add(clase)
            conteos_clase[clase] += 1
            for i, valor in enumerate(features):
                conteos_feature[i][clase][valor] += 1
        
        # Calcular probabilidades
        for clase in self.clases:
            self.prior[clase] = conteos_clase[clase] / n
            
            for i in conteos_feature:
                total = sum(conteos_feature[i][clase].values())
                for valor in conteos_feature[i][clase]:
                    # Suavizado de Laplace
                    self.likelihood[i][clase][valor] = (
                        (conteos_feature[i][clase][valor] + 1) / (total + len(conteos_feature[i][clase]))
                    )
    
    def predecir(self, features):
        scores = {}
        for clase in self.clases:
            score = math.log(self.prior[clase])
            for i, valor in enumerate(features):
                prob = self.likelihood[i][clase].get(valor, 1e-10)
                score += math.log(prob)
            scores[clase] = score
        return max(scores, key=scores.get)

print("Naïve Bayes: Clasificador simple y eficiente")
print("Asume independencia condicional de features")
