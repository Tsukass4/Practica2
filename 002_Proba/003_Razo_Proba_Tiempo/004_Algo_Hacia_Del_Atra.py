"""
Algoritmo 55: Algoritmo Hacia Delante-Atrás (Forward-Backward)
Algoritmo de suavizado que combina pases hacia adelante y atrás.
"""
class AlgoritmoAdelanteAtras:
    def __init__(self, transicion, observacion):
        self.T = transicion
        self.O = observacion
    
    def adelante(self, evidencias, prior):
        # Pase hacia adelante: calcular α
        alphas = [prior]
        for e in evidencias:
            alpha = {}
            for x in self.T:
                alpha[x] = self.O[x][e] * sum(
                    self.T[x_prev][x] * alphas[-1].get(x_prev, 0)
                    for x_prev in self.T
                )
            # Normalizar
            total = sum(alpha.values())
            alphas.append({k: v/total for k, v in alpha.items()})
        return alphas
    
    def atras(self, evidencias):
        # Pase hacia atrás: calcular β
        betas = [{x: 1.0 for x in self.T}]
        for e in reversed(evidencias[1:]):
            beta = {}
            for x in self.T:
                beta[x] = sum(
                    self.T[x][x_next] * self.O[x_next][e] * betas[0].get(x_next, 0)
                    for x_next in self.T
                )
            betas.insert(0, beta)
        return betas
    
    def suavizado(self, evidencias, prior):
        alphas = self.adelante(evidencias, prior)
        betas = self.atras(evidencias)
        # Combinar α y β
        suavizados = []
        for alpha, beta in zip(alphas, betas):
            sv = {x: alpha[x] * beta[x] for x in alpha}
            total = sum(sv.values())
            suavizados.append({x: v/total for x, v in sv.items()})
        return suavizados

print("Algoritmo Adelante-Atrás: Suavizado óptimo")
print("Combina información pasada (α) y futura (β)")
