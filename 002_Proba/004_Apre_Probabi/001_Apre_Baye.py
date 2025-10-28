"""
Algoritmo 60: Aprendizaje Bayesiano
Actualización de creencias sobre hipótesis usando datos.
P(h|D) = P(D|h) * P(h) / P(D)
"""
class AprendizajeBayesiano:
    def __init__(self, hipotesis, prior):
        self.hipotesis = hipotesis
        self.posterior = prior.copy()
    
    def actualizar(self, datos, verosimilitud):
        # P(h|D) ∝ P(D|h) * P(h)
        for h in self.hipotesis:
            self.posterior[h] *= verosimilitud(datos, h)
        
        # Normalizar
        total = sum(self.posterior.values())
        self.posterior = {h: p/total for h, p in self.posterior.items()}
    
    def prediccion(self, x, prob_prediccion):
        # P(x|D) = Σ P(x|h) * P(h|D)
        return sum(
            prob_prediccion(x, h) * self.posterior[h]
            for h in self.hipotesis
        )

# Ejemplo
print("Aprendizaje Bayesiano: Actualización de creencias con datos")
print("Combina prior con verosimilitud para obtener posterior")
