'''Algoritmo 46: Manto de Markov
El manto de Markov de un nodo X son sus padres, hijos y padres de sus hijos.
X es condicionalmente independiente del resto dado su manto de Markov.
'''
class MantoMarkov:
    @staticmethod
    def obtener_manto(nodo, red):
        manto = set()
        manto.update(nodo.padres)
        manto.update(nodo.hijos)
        for hijo in nodo.hijos:
            manto.update(hijo.padres)
        manto.discard(nodo)
        return manto

# Ejemplo
print("Manto de Markov: Conjunto mínimo que hace a X independiente del resto")
print("Manto(X) = Padres(X) ∪ Hijos(X) ∪ Padres(Hijos(X))")
