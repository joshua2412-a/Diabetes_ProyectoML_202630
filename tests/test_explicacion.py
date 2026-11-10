"""Pruebas de la explicación local por valores de Shapley (src/explicacion.py)."""

import numpy as np
import pandas as pd
import pytest

from src.explicacion import shapley_muestra, shapley_permutacion


@pytest.fixture
def tabla():
    aleatorio = np.random.default_rng(0)
    return pd.DataFrame({
        "a": aleatorio.normal(size=40),
        "b": aleatorio.integers(0, 5, size=40),
        "c": aleatorio.choice(["X", "Y", "Z"], size=40),
        "muda": aleatorio.normal(size=40),
    })


def _no_lineal(t):
    efecto_c = t["c"].map({"X": 0.0, "Y": 0.8, "Z": -0.5}).to_numpy()
    z = t["a"].to_numpy() * t["b"].to_numpy() + efecto_c
    return 1 / (1 + np.exp(-z))


def test_eficiencia_exacta(tabla):
    """Las contribuciones suman f(caso) - E[f(fondo)] aun con pocas permutaciones."""
    caso, fondo = tabla.iloc[[0]], tabla.iloc[1:21]
    r = shapley_permutacion(_no_lineal, caso, fondo, n_permutaciones=2)
    assert r["contribuciones"].sum() == pytest.approx(
        _no_lineal(caso)[0] - _no_lineal(fondo).mean(), abs=1e-12)
    assert r["prediccion"] == pytest.approx(_no_lineal(caso)[0])


def test_variable_muda_recibe_cero(tabla):
    r = shapley_permutacion(_no_lineal, tabla.iloc[[3]], tabla.iloc[5:25])
    assert r["contribuciones"]["muda"] == 0.0


def test_modelo_lineal_es_exacto(tabla):
    """En un modelo lineal, phi_j = beta_j (x_j - media del fondo)."""
    beta = {"a": 2.0, "b": -0.5, "muda": 0.3}
    lineal = lambda t: sum(beta[k] * t[k].to_numpy() for k in beta)
    caso, fondo = tabla.iloc[[1]], tabla.iloc[10:30]
    r = shapley_permutacion(lineal, caso, fondo, n_permutaciones=4)
    for k, coef in beta.items():
        esperado = coef * (caso[k].iloc[0] - fondo[k].mean())
        assert r["contribuciones"][k] == pytest.approx(esperado, abs=1e-12)
    assert r["contribuciones"]["c"] == pytest.approx(0.0, abs=1e-12)


def test_misma_semilla_mismo_resultado(tabla):
    a = shapley_permutacion(_no_lineal, tabla.iloc[[2]], tabla.iloc[5:25],
                            semilla=7)
    b = shapley_permutacion(_no_lineal, tabla.iloc[[2]], tabla.iloc[5:25],
                            semilla=7)
    pd.testing.assert_series_equal(a["contribuciones"], b["contribuciones"])


def test_una_sola_llamada_al_modelo(tabla):
    llamadas = []

    def contar(t):
        llamadas.append(len(t))
        return _no_lineal(t)

    shapley_permutacion(contar, tabla.iloc[[0]], tabla.iloc[1:11],
                        n_permutaciones=4)
    assert llamadas == [4 * (tabla.shape[1] + 1) * 10]


def test_muestra_devuelve_una_fila_por_caso(tabla):
    r = shapley_muestra(_no_lineal, tabla.iloc[:3], tabla.iloc[10:20])
    assert r.shape == (3, tabla.shape[1])
