"""
Algoritmo 77: Gramáticas Probabilísticas Independientes del Contexto (PCFG)
Gramáticas con probabilidades para parsing.
"""
class PCFG:
    def __init__(self):
        self.reglas = {}  # {no_terminal: [(produccion, probabilidad)]}
    
    def agregar_regla(self, no_terminal, produccion, probabilidad):
        if no_terminal not in self.reglas:
            self.reglas[no_terminal] = []
        self.reglas[no_terminal].append((produccion, probabilidad))
    
    def normalizar(self):
        for nt in self.reglas:
            total = sum(p for _, p in self.reglas[nt])
            self.reglas[nt] = [(prod, p/total) for prod, p in self.reglas[nt]]
    
    def generar(self, simbolo='S', max_profundidad=10):
        if max_profundidad == 0:
            return ""
        
        if simbolo not in self.reglas:
            return simbolo
        
        import random
        producciones, probs = zip(*self.reglas[simbolo])
        produccion = random.choices(producciones, weights=probs)[0]
        
        resultado = []
        for simbolo_prod in produccion:
            resultado.append(self.generar(simbolo_prod, max_profundidad - 1))
        
        return ' '.join(resultado)

# Ejemplo
pcfg = PCFG()
pcfg.agregar_regla('S', ['NP', 'VP'], 1.0)
pcfg.agregar_regla('NP', ['Det', 'N'], 0.6)
pcfg.agregar_regla('NP', ['N'], 0.4)
pcfg.agregar_regla('VP', ['V', 'NP'], 1.0)
pcfg.agregar_regla('Det', ['el'], 0.5)
pcfg.agregar_regla('Det', ['la'], 0.5)
pcfg.agregar_regla('N', ['gato'], 0.5)
pcfg.agregar_regla('N', ['perro'], 0.5)
pcfg.agregar_regla('V', ['come'], 1.0)

print("PCFG: Gramática Probabilística")
print(f"Oración generada: {pcfg.generar()}")
