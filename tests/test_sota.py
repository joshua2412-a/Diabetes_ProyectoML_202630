"""Pruebas de los benchmarks del estado del arte (src/sota.py)."""

import numpy as np
import pytest
from sklearn.base import clone

from src.espacios import muestrear
from src.sota import ESPACIOS, MODELOS_SOTA, construir_pipeline_sota, crear_modelo_sota


@pytest.mark.parametrize("nombre", MODELOS_SOTA)
def test_espacio_valido_y_ajuste(nombre, datos, roles):
    pytest.importorskip(nombre)
    X, y, _ = datos
    aleatorio = np.random.default_rng(0)
    parametros = {c: muestrear(d, aleatorio) for c, d in ESPACIOS[nombre].items()}
    pipeline = clone(construir_pipeline_sota(nombre, roles)).set_params(**parametros)
    p = pipeline.fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("nombre", MODELOS_SOTA)
def test_misma_semilla_mismas_predicciones(nombre, datos, roles):
    pytest.importorskip(nombre)
    X, y, _ = datos
    a = construir_pipeline_sota(nombre, roles).fit(X, y).predict_proba(X)
    b = construir_pipeline_sota(nombre, roles).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b)


def test_benchmark_desconocido():
    with pytest.raises(ValueError):
        crear_modelo_sota("tabpfn")
