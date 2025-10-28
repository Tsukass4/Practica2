"""
Algoritmo 78: Gramáticas Probabilísticas Lexicalizadas
PCFG con información léxica para mejor parsing.
"""
class PCFGLexicalizada:
    def __init__(self):
        self.reglas = {}  # {(no_terminal, head_word): [(produccion, prob)]}
    
    def agregar_regla(self, no_terminal, head_word, produccion, probabilidad):
        clave = (no_terminal, head_word)
        if clave not in self.reglas:
            self.reglas[clave] = []
        self.reglas[clave].append((produccion, probabilidad))
    
    def probabilidad_regla(self, no_terminal, head_word, produccion):
        clave = (no_terminal, head_word)
        if clave not in self.reglas:
            return 0.0
        
        for prod, prob in self.reglas[clave]:
            if prod == produccion:
                return prob
        return 0.0

print("PCFG Lexicalizada: Incorpora información de palabras principales")
print("Mejora desambiguación en parsing")
