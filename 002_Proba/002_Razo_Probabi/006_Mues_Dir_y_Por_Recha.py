'''Algoritmo 49: Muestreo Directo y Por Rechazo
Métodos de inferencia aproximada usando simulación Monte Carlo.
'''
import random

def muestreo_directo(red, n_muestras=1000):
    muestras = []
    for _ in range(n_muestras):
        muestra = {}
        for nodo in red.orden_topologico():
            valores_padres = tuple(muestra[p] for p in nodo.padres)
            # Muestrear según P(nodo | padres)
            probs = nodo.cpt[valores_padres]
            muestra[nodo] = random.choices(nodo.valores, weights=probs.values())[0]
        muestras.append(muestra)
    return muestras

def muestreo_por_rechazo(consulta, evidencia, red, n_muestras=10000):
    muestras_validas = []
    for _ in range(n_muestras):
        muestra = muestreo_directo(red, 1)[0]
        # Rechazar si no coincide con evidencia
        if all(muestra[var] == val for var, val in evidencia.items()):
            muestras_validas.append(muestra[consulta])
    
    # Estimar probabilidades
    from collections import Counter
    conteos = Counter(muestras_validas)
    total = len(muestras_validas)
    return {val: count/total for val, count in conteos.items()}

print("Muestreo Directo: Genera muestras de la distribución conjunta")
print("Muestreo por Rechazo: Filtra muestras que coinciden con evidencia")
print("Problema: Ineficiente si evidencia es improbable")
