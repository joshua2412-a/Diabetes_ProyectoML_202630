"""Pruebas de los espacios de búsqueda (src/espacios.py)."""

import numpy as np
import pytest

from src.espacios import (discretizar, espacio_busqueda, muestrear,
                          presupuesto_modelo, rejilla_hiperparametros)
from src.modelos import MODELOS


def test_presupuestos_del_entregable_2():
    """El presupuesto del E2 es el tamaño de la rejilla: 8 en los modelos
    costosos y 12 en los económicos (sección 8.2)."""
    esperado = {"knn": 8, "bayes": 12, "logistica": 12, "arbol": 12,
                "random_forest": 8, "xgboost": 8, "svm": 12}
    assert {m: presupuesto_modelo(m) for m in MODELOS} == esperado


@pytest.mark.parametrize("modelo", MODELOS)
def test_muestras_dentro_del_espacio(modelo):
    aleatorio = np.random.default_rng(0)
    for clave, distribucion in espacio_busqueda(modelo).items():
        for _ in range(50):
            valor = muestrear(distribucion, aleatorio)
            if distribucion[0] == "cat":
                assert valor in distribucion[1]
            else:
                assert distribucion[1] <= valor <= distribucion[2], clave


def test_discretizacion_continua_en_16_niveles():
    niveles = discretizar(("log", 1e-3, 1e2))
    assert len(niveles) == 16
    assert niveles[0] == pytest.approx(1e-3) and niveles[-1] == pytest.approx(1e2)


def test_modelo_sin_espacio():
    with pytest.raises(ValueError):
        rejilla_hiperparametros("inventado")
    with pytest.raises(ValueError):
        espacio_busqueda("inventado")
