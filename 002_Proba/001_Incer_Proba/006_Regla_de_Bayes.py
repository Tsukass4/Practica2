"""
Algoritmo 43: Regla de Bayes

La Regla de Bayes permite actualizar probabilidades cuando obtenemos nueva evidencia.

Fórmula:
P(H|E) = P(E|H) * P(H) / P(E)

Donde:
- P(H|E): Probabilidad posterior (después de observar E)
- P(E|H): Verosimilitud (likelihood)
- P(H): Probabilidad a priori
- P(E): Evidencia (constante de normalización)

Forma extendida:
P(H|E) = P(E|H) * P(H) / [P(E|H)*P(H) + P(E|¬H)*P(¬H)]
"""

from typing import Dict, List
import math


class ReglaBayes:
    """Implementación de la Regla de Bayes"""
    
    @staticmethod
    def posterior(prior: float, likelihood: float, evidencia: float) -> float:
        """
        Calcula P(H|E) = P(E|H) * P(H) / P(E)
        
        Args:
            prior: P(H)
            likelihood: P(E|H)
            evidencia: P(E)
        """
        if evidencia == 0:
            return 0
        return (likelihood * prior) / evidencia
    
    @staticmethod
    def posterior_binario(prior_h: float, likelihood_h: float, likelihood_not_h: float) -> float:
        """
        Calcula P(H|E) para hipótesis binaria
        
        P(H|E) = P(E|H)*P(H) / [P(E|H)*P(H) + P(E|¬H)*P(¬H)]
        """
        prior_not_h = 1 - prior_h
        evidencia = likelihood_h * prior_h + likelihood_not_h * prior_not_h
        
        if evidencia == 0:
            return 0
        
        return (likelihood_h * prior_h) / evidencia
    
    @staticmethod
    def posterior_multiple(priors: Dict[str, float], likelihoods: Dict[str, float]) -> Dict[str, float]:
        """
        Calcula P(H_i|E) para múltiples hipótesis
        
        P(H_i|E) = P(E|H_i)*P(H_i) / Σ_j P(E|H_j)*P(H_j)
        """
        # Calcular numeradores
        numeradores = {h: likelihoods[h] * priors[h] for h in priors}
        
        # Calcular evidencia (denominador)
        evidencia = sum(numeradores.values())
        
        if evidencia == 0:
            return {h: 0 for h in priors}
        
        # Calcular posteriores
        posteriores = {h: num / evidencia for h, num in numeradores.items()}
        
        return posteriores


# Ejemplo 1: Test médico
def ejemplo_test_medico():
    """Ejemplo clásico: Test de enfermedad"""
    print("=== Regla de Bayes: Test Médico ===\n")
    
    print("Escenario:")
    print("  - Enfermedad rara: 1% de la población la tiene")
    print("  - Test con 95% de sensibilidad (detecta enfermedad si existe)")
    print("  - Test con 90% de especificidad (negativo si no hay enfermedad)")
    print()
    
    # Datos
    prior_enfermo = 0.01  # P(Enfermo)
    sensibilidad = 0.95   # P(Test+ | Enfermo)
    especificidad = 0.90  # P(Test- | Sano)
    
    # P(Test+ | Sano) = 1 - especificidad
    prob_falso_positivo = 1 - especificidad  # 0.10
    
    print("Pregunta: Si el test da positivo, ¿cuál es la probabilidad de estar enfermo?")
    print()
    
    # Aplicar Regla de Bayes
    posterior = ReglaBayes.posterior_binario(
        prior_h=prior_enfermo,
        likelihood_h=sensibilidad,
        likelihood_not_h=prob_falso_positivo
    )
    
    print("Solución usando Regla de Bayes:")
    print(f"  P(Enfermo) = {prior_enfermo:.3f} (prior)")
    print(f"  P(Test+ | Enfermo) = {sensibilidad:.2f} (likelihood)")
    print(f"  P(Test+ | Sano) = {prob_falso_positivo:.2f}")
    print()
    print(f"  P(Enfermo | Test+) = {posterior:.4f} = {posterior*100:.2f}%")
    print()
    
    print("Interpretación:")
    print(f"  - Aunque el test es 95% preciso, solo hay {posterior*100:.1f}% de probabilidad")
    print(f"    de estar enfermo con test positivo")
    print(f"  - Esto se debe a que la enfermedad es rara (prior bajo)")
    print(f"  - Muchos falsos positivos debido a la gran población sana")


# Ejemplo 2: Filtro de spam
def ejemplo_filtro_spam():
    """Ejemplo: Clasificación de spam con Bayes"""
    print("\n\n=== Regla de Bayes: Filtro de Spam ===\n")
    
    print("Datos de entrenamiento:")
    print("  - 30% de emails son spam")
    print("  - Palabra 'oferta' aparece en:")
    print("    • 80% de emails spam")
    print("    • 10% de emails legítimos")
    print()
    
    # Priors
    prior_spam = 0.30
    prior_legitimo = 0.70
    
    # Likelihoods
    prob_oferta_dado_spam = 0.80
    prob_oferta_dado_legitimo = 0.10
    
    print("Pregunta: Si un email contiene 'oferta', ¿es spam?")
    print()
    
    # Aplicar Bayes
    posterior_spam = ReglaBayes.posterior_binario(
        prior_h=prior_spam,
        likelihood_h=prob_oferta_dado_spam,
        likelihood_not_h=prob_oferta_dado_legitimo
    )
    
    print("Solución:")
    print(f"  P(Spam | 'oferta') = {posterior_spam:.4f} = {posterior_spam*100:.1f}%")
    print(f"  P(Legítimo | 'oferta') = {1-posterior_spam:.4f} = {(1-posterior_spam)*100:.1f}%")
    print()
    print(f"Decisión: Clasificar como {'SPAM' if posterior_spam > 0.5 else 'LEGÍTIMO'}")


# Ejemplo 3: Diagnóstico con múltiples enfermedades
def ejemplo_diagnostico_multiple():
    """Ejemplo: Diagnóstico con múltiples hipótesis"""
    print("\n\n=== Regla de Bayes: Diagnóstico Múltiple ===\n")
    
    print("Paciente con fiebre alta")
    print()
    
    # Priors (prevalencia de enfermedades)
    priors = {
        "gripe": 0.40,
        "resfriado": 0.35,
        "neumonia": 0.05,
        "otras": 0.20
    }
    
    # Likelihoods P(Fiebre Alta | Enfermedad)
    likelihoods = {
        "gripe": 0.80,
        "resfriado": 0.20,
        "neumonia": 0.95,
        "otras": 0.30
    }
    
    print("Probabilidades a Priori (prevalencia):")
    for enfermedad, prob in priors.items():
        print(f"  P({enfermedad:12s}) = {prob:.2f}")
    
    print("\nProbabilidad de fiebre alta dada cada enfermedad:")
    for enfermedad, prob in likelihoods.items():
        print(f"  P(Fiebre Alta | {enfermedad:12s}) = {prob:.2f}")
    
    # Aplicar Bayes
    posteriores = ReglaBayes.posterior_multiple(priors, likelihoods)
    
    print("\nProbabilidades Posteriores (después de observar fiebre alta):")
    posteriores_ordenados = sorted(posteriores.items(), key=lambda x: x[1], reverse=True)
    
    for enfermedad, prob in posteriores_ordenados:
        barra = '█' * int(prob * 50)
        print(f"  P({enfermedad:12s} | Fiebre Alta) = {prob:.4f} {barra}")
    
    print(f"\nDiagnóstico más probable: {posteriores_ordenados[0][0]}")


# Ejemplo 4: Actualización secuencial
def ejemplo_actualizacion_secuencial():
    """Ejemplo: Actualización bayesiana secuencial"""
    print("\n\n=== Regla de Bayes: Actualización Secuencial ===\n")
    
    print("Moneda posiblemente cargada")
    print("  - Prior: 50% justa, 50% cargada (80% caras)")
    print()
    
    # Prior inicial
    prob_cargada = 0.5
    
    # Lanzamientos observados
    lanzamientos = ["cara", "cara", "cruz", "cara", "cara", "cara"]
    
    print("Lanzamientos observados:")
    for i, resultado in enumerate(lanzamientos, 1):
        # Likelihoods
        if resultado == "cara":
            likelihood_cargada = 0.80
            likelihood_justa = 0.50
        else:  # cruz
            likelihood_cargada = 0.20
            likelihood_justa = 0.50
        
        # Actualizar usando Bayes
        prob_cargada = ReglaBayes.posterior_binario(
            prior_h=prob_cargada,
            likelihood_h=likelihood_cargada,
            likelihood_not_h=likelihood_justa
        )
        
        print(f"  {i}. {resultado:5s} → P(Cargada) = {prob_cargada:.4f}")
    
    print(f"\nConclusión: {prob_cargada*100:.1f}% de probabilidad de que esté cargada")


# Ejecutar ejemplos
if __name__ == "__main__":
    print("=== Regla de Bayes ===\n")
    
    ejemplo_test_medico()
    ejemplo_filtro_spam()
    ejemplo_diagnostico_multiple()
    ejemplo_actualizacion_secuencial()
    
    print("\n" + "="*70)
    print("\nRegla de Bayes:")
    print("  P(H|E) = P(E|H) * P(H) / P(E)")
    print("\nComponentes:")
    print("  - P(H|E): Posterior (lo que queremos)")
    print("  - P(E|H): Likelihood (verosimilitud)")
    print("  - P(H): Prior (conocimiento previo)")
    print("  - P(E): Evidencia (normalización)")
    print("\nAplicaciones:")
    print("  - Diagnóstico médico")
    print("  - Filtrado de spam")
    print("  - Clasificación de documentos")
    print("  - Inferencia científica")

