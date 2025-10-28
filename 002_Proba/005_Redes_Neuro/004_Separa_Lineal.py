"""
Algoritmo 71: Separabilidad Lineal
Concepto de qué problemas puede resolver un perceptrón.
"""
import numpy as np

class SeparabilidadLineal:
    @staticmethod
    def es_linealmente_separable(X, y):
        # Intenta encontrar hiperplano separador
        from sklearn.svm import LinearSVC
        try:
            clf = LinearSVC()
            clf.fit(X, y)
            return clf.score(X, y) == 1.0
        except:
            return False
    
    @staticmethod
    def ejemplos():
        print("Problemas Linealmente Separables:")
        print("  ✓ AND, OR, NOT")
        print("  ✓ Clasificación binaria simple")
        print()
        print("Problemas NO Linealmente Separables:")
        print("  ✗ XOR")
        print("  ✗ Requieren redes multicapa")
        
        # XOR
        X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
        y_xor = np.array([0, 1, 1, 0])
        print(f"\nXOR es separable: {SeparabilidadLineal.es_linealmente_separable(X_xor, y_xor)}")

SeparabilidadLineal.ejemplos()
