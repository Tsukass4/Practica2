"""
Algoritmo 54: Filtrado, Predicción, Suavizado y Explicación
- Filtrado: P(Xt | e1:t) - estado actual dada evidencia pasada
- Predicción: P(Xt+k | e1:t) - estado futuro
- Suavizado: P(Xk | e1:t) - estado pasado (k < t)
- Explicación: argmax P(x1:t | e1:t) - secuencia más probable
"""
class InferenciaTemporal:
    @staticmethod
    def filtrado(creencia_anterior, transicion, observacion, evidencia):
        # P(Xt | e1:t) = α P(et | Xt) Σ P(Xt | xt-1) P(xt-1 | e1:t-1)
        prediccion = {}
        for xt in transicion:
            prediccion[xt] = sum(
                transicion[xt_1][xt] * creencia_anterior.get(xt_1, 0)
                for xt_1 in transicion
            )
        # Actualizar con observación
        creencia = {x: observacion[x][evidencia] * prediccion[x] for x in prediccion}
        # Normalizar
        total = sum(creencia.values())
        return {x: p/total for x, p in creencia.items()}
    
    @staticmethod
    def prediccion(creencia, transicion, pasos):
        # P(Xt+k | e1:t)
        for _ in range(pasos):
            nueva_creencia = {}
            for xt in transicion:
                nueva_creencia[xt] = sum(
                    transicion[xt_1][xt] * creencia.get(xt_1, 0)
                    for xt_1 in transicion
                )
            creencia = nueva_creencia
        return creencia

print("Filtrado: Estimar estado actual")
print("Predicción: Estimar estado futuro")
print("Suavizado: Estimar estado pasado con toda la evidencia")
