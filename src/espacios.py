"""
Espacios de búsqueda de hiperparámetros y presupuesto de cada modelo.

Forma parte de ``src/``, la versión modular del motor del experimento. Las
funciones se trasladaron sin cambios de lógica desde ``docs/experimento.py``,
que se conserva congelado para reproducir el Entregable 2; la equivalencia
entre ambos se verifica en ``tests/test_equivalencia_e2.py``.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import ParameterGrid

# Cada modelo tiene dos descripciones del mismo espacio:
#
# * Una REJILLA gruesa (listas de valores), que Grid Search recorre de forma
#   exhaustiva. Su tamaño fija el presupuesto de la corrida: 12
#   configuraciones en los modelos económicos y 8 en los costosos.
# * Un ESPACIO CONTINUO (distribuciones), del que muestrean Random Search,
#   Optuna y el algoritmo genético con ese mismo presupuesto. Es lo que
#   permite a Random Search probar valores que la rejilla nunca visita
#   (Bergstra y Bengio, 2012) y a Optuna modelar la superficie de respuesta.
#
# Formato de las distribuciones:
#   ("log", a, b)      real log-uniforme en [a, b]
#   ("float", a, b)    real uniforme en [a, b]
#   ("int", a, b)      entero uniforme en [a, b]
#   ("intlog", a, b)   entero log-uniforme en [a, b]
#   ("cat", [...])     categórica

_REJILLAS = {
    # k grande suaviza el voto; con 11 % de positivos, k pequeño deja
    # vecindarios sin casos positivos. weights="distance" da más peso a los
    # vecinos cercanos.
    "knn": {"modelo__n_neighbors": [25, 51, 101, 201],
            "modelo__weights": ["uniform", "distance"]},
    # var_smoothing estabiliza las varianzas casi nulas de las variables
    # one-hot poco frecuentes; recorre doce órdenes de magnitud.
    "bayes": {"modelo__var_smoothing": list(np.logspace(-11, 0, 12))},
    "logistica": {"modelo__C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
                  "modelo__penalty": ["l1", "l2"]},
    "arbol": {"modelo__max_depth": [4, 8, 16],
              "modelo__min_samples_leaf": [20, 100],
              "modelo__criterion": ["gini", "entropy"]},
    "random_forest": {"modelo__max_depth": [8, 16],
                      "modelo__min_samples_leaf": [5, 25],
                      "modelo__max_features": ["sqrt", 0.3]},
    "xgboost": {"modelo__max_depth": [3, 6],
                "modelo__learning_rate": [0.05, 0.2],
                "modelo__min_child_weight": [1, 10]},
    # El prefijo estimator__ atraviesa el envoltorio de calibración.
    "svm": {"modelo__estimator__C": list(np.logspace(-5, 2, 12))},
}


_DISTRIBUCIONES = {
    "knn": {"modelo__n_neighbors": ("intlog", 5, 301),
            "modelo__weights": ("cat", ["uniform", "distance"]),
            "modelo__p": ("cat", [1, 2])},
    "bayes": {"modelo__var_smoothing": ("log", 1e-12, 1.0)},
    "logistica": {"modelo__C": ("log", 1e-4, 1e2),
                  "modelo__penalty": ("cat", ["l1", "l2"])},
    "arbol": {"modelo__max_depth": ("int", 2, 30),
              "modelo__min_samples_leaf": ("intlog", 1, 500),
              "modelo__criterion": ("cat", ["gini", "entropy"]),
              "modelo__ccp_alpha": ("log", 1e-7, 1e-2)},
    "random_forest": {"modelo__n_estimators": ("int", 100, 500),
                      "modelo__max_depth": ("int", 4, 30),
                      "modelo__min_samples_leaf": ("intlog", 1, 100),
                      "modelo__max_features": ("float", 0.05, 0.6)},
    "xgboost": {"modelo__learning_rate": ("log", 0.01, 0.3),
                "modelo__max_depth": ("int", 2, 10),
                "modelo__min_child_weight": ("log", 1.0, 50.0),
                "modelo__subsample": ("float", 0.5, 1.0),
                "modelo__colsample_bytree": ("float", 0.5, 1.0)},
    "svm": {"modelo__estimator__C": ("log", 1e-6, 1e2)},
}


def rejilla_hiperparametros(nombre):
    """Rejilla gruesa que recorre Grid Search de forma exhaustiva.

    Returns
    -------
    dict
        Nombre del hiperparámetro (con prefijo ``modelo__``) a lista de
        valores.
    """
    try:
        return _REJILLAS[nombre]
    except KeyError:
        raise ValueError(f"Sin rejilla definida para: {nombre}")


def espacio_busqueda(nombre):
    """Espacio continuo del que muestrean Random, Optuna y el genético.

    Returns
    -------
    dict
        Nombre del hiperparámetro a distribución (véase el formato en el
        comentario de la sección).
    """
    try:
        return _DISTRIBUCIONES[nombre]
    except KeyError:
        raise ValueError(f"Sin espacio definido para: {nombre}")


def presupuesto_modelo(nombre):
    """Número de configuraciones por corrida: el tamaño de la rejilla.

    Los cuatro optimizadores de un mismo modelo evalúan exactamente este
    número de configuraciones, de modo que la comparación entre ellos sea a
    igual presupuesto.
    """
    return len(ParameterGrid(rejilla_hiperparametros(nombre)))


def muestrear(distribucion, aleatorio):
    """Extrae un valor de una distribución del espacio de búsqueda."""
    tipo = distribucion[0]
    if tipo == "cat":
        opciones = distribucion[1]
        return opciones[int(aleatorio.integers(0, len(opciones)))]
    a, b = distribucion[1], distribucion[2]
    if tipo == "log":
        return float(np.exp(aleatorio.uniform(np.log(a), np.log(b))))
    if tipo == "float":
        return float(aleatorio.uniform(a, b))
    if tipo == "int":
        return int(aleatorio.integers(a, b + 1))
    if tipo == "intlog":
        return int(round(np.exp(aleatorio.uniform(np.log(a), np.log(b)))))
    raise ValueError(f"Distribución no reconocida: {distribucion}")


def discretizar(distribucion, niveles=16):
    """Discretiza una distribución para el genoma del algoritmo genético."""
    tipo = distribucion[0]
    if tipo == "cat":
        return list(distribucion[1])
    a, b = distribucion[1], distribucion[2]
    if tipo == "log":
        return [float(v) for v in np.geomspace(a, b, niveles)]
    if tipo == "float":
        return [float(v) for v in np.linspace(a, b, niveles)]
    if tipo == "int":
        return sorted({int(round(v)) for v in np.linspace(a, b, niveles)})
    if tipo == "intlog":
        return sorted({int(round(v)) for v in np.geomspace(a, b, niveles)})
    raise ValueError(f"Distribución no reconocida: {distribucion}")
