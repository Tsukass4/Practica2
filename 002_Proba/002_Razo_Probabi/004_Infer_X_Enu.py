'''Algoritmo 47: Inferencia por Enumeración
Método exacto para calcular P(X|e) enumerando todas las asignaciones.
P(X|e) = α Σ_y P(X, e, y) donde y son variables ocultas
'''
def inferencia_enumeracion(variable, evidencia, red):
    from itertools import product
    resultados = {}
    for valor in variable.valores:
        suma = 0
        # Enumerar todas las asignaciones de variables ocultas
        ocultas = [v for v in red.variables if v != variable and v not in evidencia]
        for asignacion_ocultas in product(*[v.valores for v in ocultas]):
            asignacion_completa = evidencia.copy()
            asignacion_completa[variable] = valor
            for v, val in zip(ocultas, asignacion_ocultas):
                asignacion_completa[v] = val
            suma += red.prob_conjunta(asignacion_completa)
        resultados[valor] = suma
    # Normalizar
    total = sum(resultados.values())
    return {k: v/total for k, v in resultados.items()}

print("Inferencia por Enumeración: Método exacto pero exponencial")
print("Complejidad: O(n * 2^n) donde n = número de variables")
