"""
Algoritmo 53: Hipótesis de Markov - Procesos de Markov
El futuro es independiente del pasado dado el presente.
P(Xt | X0, ..., Xt-1) = P(Xt | Xt-1)
"""
class ProcesoMarkov:
    def __init__(self, estados, matriz_transicion, estado_inicial):
        self.estados = estados
        self.T = matriz_transicion
        self.estado = estado_inicial
    
    def siguiente_estado(self):
        import random
        probs = self.T[self.estado]
        self.estado = random.choices(self.estados, weights=probs)[0]
        return self.estado
    
    def simular(self, pasos):
        trayectoria = [self.estado]
        for _ in range(pasos):
            trayectoria.append(self.siguiente_estado())
        return trayectoria

print("Hipótesis de Markov: El futuro depende solo del presente")
print("P(Xt+1 | Xt, Xt-1, ..., X0) = P(Xt+1 | Xt)")
