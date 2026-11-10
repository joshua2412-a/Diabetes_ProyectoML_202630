"""Pruebas de calibración y predicción conforme (src/incertidumbre.py)."""

import numpy as np
import pytest

from src.incertidumbre import (conjuntos_prediccion, cuantiles_conformes,
                               error_calibracion, evaluar_conformal)


def test_ece_cero_si_calibrado_y_positivo_si_no():
    aleatorio = np.random.default_rng(0)
    p = aleatorio.uniform(0, 1, 200_000)
    y = (aleatorio.uniform(size=p.size) < p).astype(int)
    assert error_calibracion(y, p) < 0.01
    assert error_calibracion(y, np.clip(p + 0.2, 0, 1)) > 0.1


def test_cobertura_por_clase_cumple_el_nivel():
    """Con datos intercambiables, la cobertura de cada clase es >= 1 - alfa."""
    aleatorio = np.random.default_rng(1)
    n = 40_000
    p = aleatorio.beta(1, 7, n)
    y = (aleatorio.uniform(size=n) < p).astype(int)
    mitad = n // 2
    tabla, _ = evaluar_conformal(y[:mitad], p[:mitad], y[mitad:], p[mitad:], alfa=0.1)
    coberturas = tabla.set_index("grupo")["cobertura"]
    assert coberturas["Readmitidos"] >= 0.88
    assert coberturas["No readmitidos"] >= 0.88


def test_conjuntos_segun_cuantiles():
    q = {0: 0.3, 1: 0.6}
    c = conjuntos_prediccion(np.array([0.1, 0.5, 0.9]), q)
    # p=0.1: clase 0 (0.1 <= 0.3) y no clase 1 (0.9 > 0.6)
    np.testing.assert_array_equal(c, [[True, False], [False, True], [False, True]])


def test_cuantil_infinito_con_pocos_casos():
    q = cuantiles_conformes(np.array([0, 0, 0, 1]), np.array([0.1, 0.2, 0.3, 0.8]), alfa=0.1)
    assert q[1] == np.inf


def test_calibracion_sin_una_clase():
    with pytest.raises(ValueError):
        cuantiles_conformes(np.zeros(5), np.full(5, 0.1))
