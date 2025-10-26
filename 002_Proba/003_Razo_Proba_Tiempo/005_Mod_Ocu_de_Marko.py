"""
Algoritmo 56: Modelos Ocultos de Markov (HMM)
Proceso de Markov con estados ocultos y observaciones.
"""
class HMM:
    def __init__(self, estados, observaciones, transicion, emision, inicial):
        self.estados = estados
        self.observaciones = observaciones
        self.A = transicion  # P(s_t | s_t-1)
        self.B = emision     # P(o_t | s_t)
        self.pi = inicial    # P(s_0)
    
    def viterbi(self, obs_secuencia):
        # Encuentra la secuencia de estados más probable
        T = len(obs_secuencia)
        delta = [{} for _ in range(T)]
        psi = [{} for _ in range(T)]
        
        # Inicialización
        for s in self.estados:
            delta[0][s] = self.pi[s] * self.B[s][obs_secuencia[0]]
            psi[0][s] = None
        
        # Recursión
        for t in range(1, T):
            for s in self.estados:
                max_prob = max(
                    delta[t-1][s_prev] * self.A[s_prev][s]
                    for s_prev in self.estados
                )
                delta[t][s] = max_prob * self.B[s][obs_secuencia[t]]
                psi[t][s] = max(
                    self.estados,
                    key=lambda s_prev: delta[t-1][s_prev] * self.A[s_prev][s]
                )
        
        # Backtracking
        camino = [None] * T
        camino[-1] = max(self.estados, key=lambda s: delta[-1][s])
        for t in range(T-2, -1, -1):
            camino[t] = psi[t+1][camino[t+1]]
        
        return camino

print("HMM: Modelo con estados ocultos")
print("Algoritmo de Viterbi: Encuentra secuencia de estados más probable")
