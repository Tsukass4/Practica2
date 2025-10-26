'''Algoritmo 50: Ponderación de Verosimilitud
Muestreo ponderado que fija variables de evidencia y pondera por su probabilidad.
'''
import random

def ponderacion_verosimilitud(consulta, evidencia, red, n_muestras=10000):
    pesos = {val: 0.0 for val in consulta.valores}
    
    for _ in range(n_muestras):
        muestra = {}
        peso = 1.0
        
        for nodo in red.orden_topologico():
            valores_padres = tuple(muestra.get(p) for p in nodo.padres)
            
            if nodo in evidencia:
                # Fijar a valor de evidencia y actualizar peso
                muestra[nodo] = evidencia[nodo]
                peso *= nodo.cpt[valores_padres][evidencia[nodo]]
            else:
                # Muestrear normalmente
                probs = nodo.cpt[valores_padres]
                muestra[nodo] = random.choices(nodo.valores, weights=probs.values())[0]
        
        pesos[muestra[consulta]] += peso
    
    # Normalizar
    total = sum(pesos.values())
    return {val: peso/total for val, peso in pesos.items()}

print("Ponderación de Verosimilitud: Más eficiente que rechazo")
print("Fija evidencia y pondera muestras por P(evidencia | muestra)")
