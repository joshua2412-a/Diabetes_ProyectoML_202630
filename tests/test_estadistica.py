"""Pruebas de las pruebas estadísticas y tamaños del efecto
(src/estadistica.py), con valores conocidos."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.estadistica import (bootstrap_bca, correccion_holm, d_cohen,
                             d_pareada, delta_cliff, diferencia_critica,
                             magnitud_cliff, magnitud_cohen, prueba_delong,
                             tabla_cohen)


def test_cliff_con_separacion_total_y_sin_diferencia():
    assert delta_cliff([1, 2, 3], [4, 5, 6]) == pytest.approx(-1.0)
    assert delta_cliff([4, 5, 6], [1, 2, 3]) == pytest.approx(1.0)
    assert delta_cliff([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_umbrales_de_cliff():
    assert magnitud_cliff(0.1) == "Insignificante"
    assert magnitud_cliff(-0.2) == "Pequeño"
    assert magnitud_cliff(0.4) == "Mediano"
    assert magnitud_cliff(-0.9) == "Grande"


def test_d_de_cohen_y_correccion_de_hedges():
    # Medias 2 y 3, varianza 1 en ambas: d = -1. Con n = 3 + 3, el factor
    # de Hedges es 1 - 3 / (4 * 6 - 9) = 0.8.
    assert d_cohen([1, 2, 3], [2, 3, 4], corregir=False) == pytest.approx(-1.0)
    assert d_cohen([1, 2, 3], [2, 3, 4]) == pytest.approx(-0.8)


def test_d_pareada():
    # Diferencias 1, 0 y 2: media 1 y desviación 1.
    assert d_pareada([1, 2, 4], [0, 2, 2]) == pytest.approx(1.0)
    assert np.isnan(d_pareada([1, 2, 3], [0, 1, 2]))  # diferencias constantes


def test_umbrales_de_cohen():
    assert magnitud_cohen(0.1) == "Insignificante"
    assert magnitud_cohen(-0.3) == "Pequeño"
    assert magnitud_cohen(0.6) == "Mediano"
    assert magnitud_cohen(-1.2) == "Grande"
    assert magnitud_cohen(float("nan")) == "Indefinida"


def test_tabla_cohen_tiene_un_par_por_combinacion():
    por_fold = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 5], "c": [0, 1, 1]})
    assert len(tabla_cohen(por_fold)) == 3


def test_holm_ejemplo_de_libro():
    # Ordenados: 0.01 * 3 = 0.03; 0.03 * 2 = 0.06; 0.04 * 1 = 0.04, que la
    # monotonía eleva a 0.06.
    np.testing.assert_allclose(correccion_holm([0.01, 0.04, 0.03]),
                               [0.03, 0.06, 0.06])
    assert correccion_holm([0.6, 0.7]).max() <= 1.0


def test_delong_auc_coincide_con_sklearn_y_es_simetrica():
    aleatorio = np.random.default_rng(0)
    y = aleatorio.integers(0, 2, 300)
    a = y + aleatorio.normal(0, 1, 300)
    b = y + aleatorio.normal(0, 2, 300)
    r = prueba_delong(y, a, b)
    assert r["auc_a"] == pytest.approx(roc_auc_score(y, a))
    assert r["auc_b"] == pytest.approx(roc_auc_score(y, b))
    assert prueba_delong(y, b, a)["p"] == pytest.approx(r["p"])


def test_delong_predicciones_identicas():
    y = np.array([0, 1] * 50)
    p = np.linspace(0, 1, 100)
    r = prueba_delong(y, p, p.copy())
    assert r["diferencia"] == 0 and r["p"] == pytest.approx(1.0)


def test_delong_detecta_una_diferencia_grande():
    aleatorio = np.random.default_rng(1)
    y = np.array([0, 1] * 150)
    perfecto = y + aleatorio.uniform(0, 0.5, 300)
    azar = aleatorio.uniform(0, 1, 300)
    assert prueba_delong(y, perfecto, azar)["p"] < 1e-6


def test_diferencia_critica_del_capitulo_11():
    """7 modelos y 5 folds: la diferencia crítica reportada es 4.028."""
    assert diferencia_critica(7, 5) == pytest.approx(4.028, abs=1e-3)


def test_bootstrap_bca_contiene_la_estimacion():
    aleatorio = np.random.default_rng(2)
    grupos = np.repeat(np.arange(80), 2)
    y = aleatorio.integers(0, 2, 160)
    p = np.clip(y * 0.3 + aleatorio.uniform(0, 0.7, 160), 0, 1)
    r = bootstrap_bca(y, p, grupos, roc_auc_score, n_repeticiones=200)
    assert r["IC inferior"] <= r["estimación"] <= r["IC superior"]
    assert r["estimación"] == pytest.approx(roc_auc_score(y, p))


def test_nadeau_bengio_calculado_a_mano():
    from src.estadistica import prueba_nadeau_bengio
    d = np.array([0.004, -0.001, 0.006, 0.002, 0.003])
    t_esperado = d.mean() / np.sqrt((1 / 5 + 0.25) * d.var(ddof=1))
    r = prueba_nadeau_bengio(d)
    assert r["t"] == pytest.approx(t_esperado)
    assert r["gl"] == 4
    assert 0 < r["p"] < 0.5
    # La corrección siempre da un t menor que la t pareada ordinaria.
    assert r["t"] < d.mean() / np.sqrt(d.var(ddof=1) / 5)
    assert prueba_nadeau_bengio(-d)["p"] > 0.5


def test_mcnemar_cuenta_discordantes_y_es_simetrica():
    from src.estadistica import prueba_mcnemar_exacta
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    a = np.array([1, 1, 0, 0, 0, 0, 1, 1])   # falla en las posiciones 4 y 7
    b = np.array([1, 0, 0, 1, 1, 0, 1, 0])   # falla en las posiciones 1 y 3
    r = prueba_mcnemar_exacta(y, a, b)
    assert (r["n01"], r["n10"]) == (2, 2)
    assert r["p"] == pytest.approx(1.0)
    r2 = prueba_mcnemar_exacta(y, b, a)
    assert (r2["n01"], r2["n10"]) == (2, 2) and r2["p"] == pytest.approx(r["p"])


def test_mcnemar_binomial_exacta():
    from scipy import stats
    from src.estadistica import prueba_mcnemar_exacta
    # 10 discordantes, todos a favor de a: p = 2 * 0.5**10.
    y = np.ones(10, dtype=int)
    r = prueba_mcnemar_exacta(y, np.ones(10), np.zeros(10))
    assert (r["n01"], r["n10"]) == (10, 0)
    assert r["p"] == pytest.approx(2 * 0.5 ** 10)
    assert r["p"] == pytest.approx(stats.binomtest(0, 10, 0.5).pvalue)
    # Sin discordantes no hay evidencia.
    assert prueba_mcnemar_exacta(y, np.ones(10), np.ones(10))["p"] == 1.0


def test_t_bayesiana_suma_uno_y_coincide_con_nadeau_bengio():
    from scipy import stats
    from src.estadistica import prueba_nadeau_bengio, t_bayesiana_correlacionada
    d = np.array([0.004, -0.001, 0.006, 0.002, 0.003])
    r = t_bayesiana_correlacionada(d, rope=0.0)
    assert r["mejor"] + r["equivalente"] + r["peor"] == pytest.approx(1.0)
    # Con ROPE nula, P(peor) es el p unilateral de Nadeau-Bengio.
    assert r["peor"] == pytest.approx(prueba_nadeau_bengio(d)["p"])
    # Un ROPE más ancho solo puede aumentar la probabilidad de equivalencia.
    assert (t_bayesiana_correlacionada(d, rope=0.01)["equivalente"]
            > t_bayesiana_correlacionada(d, rope=0.001)["equivalente"])


def test_t_bayesiana_detecta_equivalencia_y_superioridad():
    from src.estadistica import t_bayesiana_correlacionada
    casi_cero = np.array([0.0005, -0.0004, 0.0003, -0.0002, 0.0001])
    assert t_bayesiana_correlacionada(casi_cero, rope=0.005)["equivalente"] > 0.99
    grande = np.array([0.030, 0.028, 0.031, 0.029, 0.032])
    assert t_bayesiana_correlacionada(grande, rope=0.005)["mejor"] > 0.99
