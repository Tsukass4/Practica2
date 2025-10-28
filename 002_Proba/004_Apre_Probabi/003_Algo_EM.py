"""
Algoritmo 62: Algoritmo EM (Expectation-Maximization)
Algoritmo iterativo para estimación de parámetros con variables latentes.
"""
import numpy as np

class AlgoritmoEM:
    def __init__(self, n_componentes):
        self.K = n_componentes
        self.mu = None
        self.sigma = None
        self.pi = None
    
    def inicializar(self, X):
        n, d = X.shape
        # Inicialización aleatoria
        self.mu = X[np.random.choice(n, self.K, replace=False)]
        self.sigma = [np.eye(d) for _ in range(self.K)]
        self.pi = np.ones(self.K) / self.K
    
    def e_step(self, X):
        # Paso E: Calcular responsabilidades
        n = len(X)
        gamma = np.zeros((n, self.K))
        
        for k in range(self.K):
            gamma[:, k] = self.pi[k] * self.gaussian(X, self.mu[k], self.sigma[k])
        
        # Normalizar
        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma
    
    def m_step(self, X, gamma):
        # Paso M: Actualizar parámetros
        n, d = X.shape
        Nk = gamma.sum(axis=0)
        
        # Actualizar pi
        self.pi = Nk / n
        
        # Actualizar mu
        self.mu = (gamma.T @ X) / Nk[:, np.newaxis]
        
        # Actualizar sigma
        for k in range(self.K):
            diff = X - self.mu[k]
            self.sigma[k] = (gamma[:, k:k+1] * diff).T @ diff / Nk[k]
    
    def gaussian(self, X, mu, sigma):
        d = len(mu)
        diff = X - mu
        return np.exp(-0.5 * np.sum(diff @ np.linalg.inv(sigma) * diff, axis=1)) /                np.sqrt((2*np.pi)**d * np.linalg.det(sigma))
    
    def ajustar(self, X, max_iter=100):
        self.inicializar(X)
        for _ in range(max_iter):
            gamma = self.e_step(X)
            self.m_step(X, gamma)

print("Algoritmo EM: Estimación con variables latentes")
print("Alterna entre E-step (expectativa) y M-step (maximización)")
