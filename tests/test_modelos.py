"""Pruebas de los modelos base (src/modelos.py)."""

import numpy as np
import pytest
from sklearn.base import clone

from src import configuracion as cfg
from src.balanceo import construir_pipeline
from src.modelos import MODELOS, XGBClasificadorParada, crear_modelo


@pytest.mark.parametrize("modelo", MODELOS)
def test_probabilidades_validas(datos, roles, modelo):
    X, y, _ = datos
    probabilidades = construir_pipeline(modelo, "ninguno", roles) \
        .fit(X, y).predict_proba(X)
    assert probabilidades.shape == (len(X), 2)
    assert np.all((probabilidades >= 0) & (probabilidades <= 1))
    np.testing.assert_allclose(probabilidades.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("modelo", ["logistica", "arbol", "random_forest",
                                    "xgboost", "svm"])
def test_misma_semilla_mismas_predicciones(datos, roles, modelo):
    X, y, _ = datos
    p1 = construir_pipeline(modelo, "ninguno", roles).fit(X, y).predict_proba(X)
    p2 = construir_pipeline(modelo, "ninguno", roles).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_clone_conserva_hiperparametros_de_xgboost():
    original = XGBClasificadorParada(max_depth=3, learning_rate=0.05,
                                     rondas_parada=10)
    copia = clone(original)
    assert copia.get_params() == original.get_params()
    assert not hasattr(copia, "modelo_")


def test_parada_temprana_recorta_arboles(datos, roles):
    X, y, _ = datos
    pipeline = construir_pipeline("xgboost", "ninguno", roles).fit(X, y)
    xgb = pipeline.named_steps["modelo"]
    assert 0 <= xgb.mejor_iteracion_ < xgb.n_estimators


def test_n_jobs_se_lee_al_crear_el_modelo():
    """Cambiar cfg.N_JOBS afecta a los modelos que se crean después.

    Es lo que permite fijar el número de procesos desde un notebook antes de
    construir los pipelines.
    """
    cfg.N_JOBS = 2
    assert crear_modelo("random_forest").n_jobs == 2
    assert crear_modelo("knn").n_jobs == 2
    assert crear_modelo("xgboost").n_jobs == 2


def test_modelo_desconocido():
    with pytest.raises(ValueError):
        crear_modelo("inventado")
