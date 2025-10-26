'''Algoritmo 48: Eliminación de Variables
Optimización de inferencia por enumeración usando factorización.
Elimina variables una por una sumando sobre sus valores.
'''
class Factor:
    def __init__(self, variables, valores):
        self.variables = variables
        self.valores = valores
    
    def producto(self, otro_factor):
        # Multiplica dos factores
        pass
    
    def sumar_variable(self, variable):
        # Suma (marginaliza) sobre una variable
        pass

def eliminacion_variables(consulta, evidencia, orden_eliminacion, red):
    factores = []
    # Crear factores iniciales desde CPTs
    for nodo in red.nodos:
        factores.append(nodo.cpt_como_factor())
    
    # Incorporar evidencia
    for variable, valor in evidencia.items():
        factores = [f.restringir(variable, valor) for f in factores]
    
    # Eliminar variables en orden
    for var in orden_eliminacion:
        factores_con_var = [f for f in factores if var in f.variables]
        factores_sin_var = [f for f in factores if var not in f.variables]
        producto = factores_con_var[0]
        for f in factores_con_var[1:]:
            producto = producto.producto(f)
        factor_marginalizado = producto.sumar_variable(var)
        factores = factores_sin_var + [factor_marginalizado]
    
    # Producto final y normalización
    resultado = factores[0]
    for f in factores[1:]:
        resultado = resultado.producto(f)
    return resultado.normalizar()

print("Eliminación de Variables: Más eficiente que enumeración")
print("Complejidad depende del orden de eliminación")
