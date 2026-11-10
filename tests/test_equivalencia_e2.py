"""
Equivalencia entre src/ y el motor congelado del Entregable 2.

``docs/experimento.py`` produjo las 112 corridas entregadas y se conserva
sin cambios para reproducirlas. ``src/`` es su versión modular, la que usa
el código nuevo. Estas pruebas garantizan que ambos dan exactamente los
mismos resultados: si alguien cambiara la lógica de ``src/``, fallarían.

La prueba rápida compara las dos implementaciones sobre datos sintéticos.
La prueba lenta reproduce con ``src/`` una corrida real de la tabla
maestra del Entregable 2, fold a fold.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

import experimento as ex
from src import configuracion as cfg
from src import validacion

from conftest import RAIZ


@pytest.fixture(autouse=True)
def motor_original_en_un_proceso():
    anterior = ex.N_JOBS
    ex.N_JOBS = 1
    yield
    ex.N_JOBS = anterior


COMBINACIONES = [
    ("logistica", "class_weight", "random", 1.0),
    ("arbol", "smote", "grid", 1.0),
    ("bayes", "ninguno", "deap", 1.0),
    ("svm", "ninguno", "optuna", 1.0),
    ("xgboost", "smote", "random", 0.5),
    ("random_forest", "ninguno", "random", 0.5),
]
# k-NN queda fuera de esta prueba: su espacio llega a 301 vecinos, más que
# las filas de un fold interno del conjunto sintético. Su equivalencia queda
# cubierta porque comparte con los demás todo el código salvo crear_modelo,
# idéntico en ambas versiones, y su ajuste se prueba en test_modelos.py.


@pytest.mark.parametrize("modelo,balanceo,optimizador,fraccion", COMBINACIONES)
def test_src_reproduce_al_motor_del_e2(datos, roles, razon, modelo, balanceo,
                                       optimizador, fraccion):
    X, y, grupos = datos
    argumentos = dict(folds_externos=3, folds_internos=2,
                      fraccion_busqueda=fraccion)
    # Presupuesto pequeño para que la prueba sea rápida; Grid recorre su
    # rejilla completa en cualquier caso.
    presupuesto = 4

    nueva = validacion.ejecutar_corrida(
        validacion.Corrida(modelo, balanceo, optimizador, presupuesto, razon),
        X, y, grupos, roles, **argumentos)
    original = ex.ejecutar_corrida(
        ex.Corrida(modelo, balanceo, optimizador, presupuesto, razon),
        X, y, grupos, roles, **argumentos)

    for metrica in ex.METRICAS:
        np.testing.assert_array_equal(
            json.loads(nueva[f"{metrica}_por_fold"]),
            json.loads(original[f"{metrica}_por_fold"]), err_msg=metrica)
    assert nueva["hiperparametros_por_fold"] == original["hiperparametros_por_fold"]
    assert nueva["n_evaluaciones"] == original["n_evaluaciones"]
    assert nueva["n_ajustes"] == original["n_ajustes"]


@pytest.mark.lento
def test_src_reproduce_una_corrida_real_de_la_tabla_maestra():
    """Naive Bayes sin balanceo con Random Search, sobre el conjunto de
    entrenamiento completo (unos 6 minutos con 4 procesos)."""
    ruta_datos = RAIZ / "data" / "processed" / "diabetes_train.parquet"
    ruta_tabla = RAIZ / "results" / "tabla_maestra_clasificacion.csv"
    if not ruta_datos.exists() or not ruta_tabla.exists():
        pytest.skip("Faltan los datos procesados o la tabla maestra")

    with open(RAIZ / "data" / "processed" / "roles_variables.json",
              encoding="utf-8") as archivo:
        roles = json.load(archivo)
    train = pd.read_parquet(ruta_datos)
    predictores = (roles["numericas"] + roles["ordinales"]
                   + roles["binarias"] + roles["categoricas"])
    X = train[predictores]
    y = train[roles["objetivo"]]
    grupos = train[roles["identificador"]]

    maestra = pd.read_csv(ruta_tabla)
    referencia = maestra.query("modelo == 'bayes' and balanceo == 'ninguno' "
                               "and optimizador == 'random'").iloc[0]

    cfg.N_JOBS = min(4, os.cpu_count() or 1)
    fila = validacion.ejecutar_corrida(
        validacion.Corrida("bayes", "ninguno", "random"),
        X, y, grupos, roles, folds_externos=5, folds_internos=3,
        fraccion_busqueda=float(referencia["fraccion_busqueda"]))

    np.testing.assert_allclose(json.loads(fila["auc_pr_por_fold"]),
                               json.loads(referencia["auc_pr_por_fold"]),
                               rtol=0, atol=1e-12)
