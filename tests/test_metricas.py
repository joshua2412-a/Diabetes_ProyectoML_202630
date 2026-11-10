"""Pruebas de las métricas (src/metricas.py) con un caso hecho a mano."""

import numpy as np
import pytest

from src.metricas import calcular_metricas


def test_valores_calculados_a_mano():
    # Orden por probabilidad: 0.8 (+), 0.4 (-), 0.35 (+), 0.1 (-).
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = calcular_metricas(y, p)

    # 3 de los 4 pares (positivo, negativo) están bien ordenados.
    assert m["auc_roc"] == pytest.approx(0.75)
    # Precisión 1 al recuperar el primer positivo y 2/3 al recuperar el
    # segundo: 0.5 * 1 + 0.5 * 2/3.
    assert m["auc_pr"] == pytest.approx(0.5 + 1 / 3)
    # Con umbral 0.5 solo se predice positivo el 0.8.
    assert m["exactitud"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["exactitud_balanceada"] == pytest.approx(0.75)
    assert m["brier"] == pytest.approx((0.01 + 0.16 + 0.4225 + 0.04) / 4)


def test_sin_positivos_predichos_no_falla():
    m = calcular_metricas(np.array([0, 1, 0, 1]), np.full(4, 0.2))
    assert m["precision"] == 0 and m["recall"] == 0
