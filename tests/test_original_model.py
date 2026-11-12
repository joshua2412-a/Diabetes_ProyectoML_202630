"""Pruebas de ``src.original_model`` con el esqueleto de la guía (sección 5.4).

Las pruebas del mecanismo de DES-CPL (región, competencia, encogimiento,
poda) están en ``tests/test_des_cpl.py``; aquí se comprueba la interfaz
pública que pide la guía.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification

from src.original_model import ModeloOriginal


@pytest.fixture
def datos():
    return make_classification(n_samples=300, n_features=6, n_informative=4, random_state=0)


def test_aprende_mejor_que_azar(datos):
    X, y = datos
    modelo = ModeloOriginal(k=10, random_state=0).fit(X, y)
    assert (modelo.predict(X) == y).mean() > 0.7


def test_reproducible_con_misma_semilla(datos):
    X, y = datos
    p1 = ModeloOriginal(k=10, random_state=1).fit(X, y).predict_proba(X)
    p2 = ModeloOriginal(k=10, random_state=1).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p1, p2)


def test_clone_conserva_hiperparametros():
    modelo = ModeloOriginal(k=15, tau=0.3, encogimiento=0.5, n_modelos=3, random_state=7)
    copia = clone(modelo)
    assert copia.get_params() == modelo.get_params()
    assert copia.semilla == 7


def test_entrada_invalida(datos):
    X, y = datos
    X_malo = X.copy()
    X_malo[0, 0] = np.nan
    with pytest.raises(ValueError):
        ModeloOriginal(k=10).fit(X_malo, y)


def test_hiperparametro_invalido(datos):
    X, y = datos
    with pytest.raises(ValueError):
        ModeloOriginal(k=0).fit(X, y)


def test_fit_no_modifica_los_estimadores(datos):
    # scikit-learn exige que fit no cambie los hiperparámetros: los modelos
    # por defecto se crean en cada ajuste, no se guardan en `estimadores`.
    X, y = datos
    modelo = ModeloOriginal(k=10).fit(X, y)
    assert modelo.estimadores is None
    assert modelo.nombres_ == ["logistica", "arbol", "bosque", "bayes"]
