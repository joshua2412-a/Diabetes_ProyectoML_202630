"""Pruebas de la tabla de líderes a igual presupuesto (60 evaluaciones) y de los
CSV que pueden faltar (src/tablero/analitica.py y datos.py).

La primera parte usa datos sintéticos; la de las cifras de control lee los CSV
de results/ y se omite si no están (en Render solo hay lo que está en Git).
"""

import numpy as np
import pandas as pd
import pytest

from src.tablero import analitica as A
from src.tablero import config as C
from src.tablero import datos as D


def _evaluacion():
    filas = []
    for fold in range(5):
        for metodo, familia, base in [("xgboost", "Modelo base", 0.23), ("bayes", "Modelo base", 0.20),
                                      ("Promedio simple", "Ensamble estático", 0.22),
                                      ("DES-CPL", "Modelo original", 0.229),
                                      ("Ablación: sin poda (7 modelos)", "Ablación", 0.228),
                                      ("DES-CPL inicial (sin poda ni encogimiento)", "Versión inicial", 0.224)]:
            filas.append({"fold": fold, "método": metodo, "familia": familia,
                          "auc_pr": base + 0.001 * fold, "auc_roc": 0.67, "brier": 0.096})
    return pd.DataFrame(filas)


def _sota():
    return pd.DataFrame([{"modelo": m, "fold": f, "auc_pr": 0.226 + 0.001 * f, "auc_roc": 0.66, "brier": 0.097}
                         for m in ("lightgbm", "catboost") for f in range(5)])


def test_tabla_deja_fuera_ablaciones_y_version_inicial():
    tabla = A.tabla_igual_presupuesto(_evaluacion(), _sota())
    assert set(tabla["clave"]) == {"xgboost", "bayes", "Promedio simple", C.MODELO_ORIGINAL,
                                   "lightgbm", "catboost"}


def test_tabla_media_y_desviacion_por_fold_con_ddof_1():
    tabla = A.tabla_igual_presupuesto(_evaluacion(), _sota()).set_index("clave")
    esperado = np.array([0.23 + 0.001 * f for f in range(5)])
    assert tabla.loc["xgboost", "auc_pr_media"] == pytest.approx(esperado.mean())
    assert tabla.loc["xgboost", "auc_pr_sd"] == pytest.approx(esperado.std(ddof=1))
    assert (tabla["n_folds"] == 5).all()


def test_tabla_ordenada_por_auc_pr_y_con_nombres_legibles():
    tabla = A.tabla_igual_presupuesto(_evaluacion(), _sota())
    assert tabla["auc_pr_media"].is_monotonic_decreasing
    assert tabla.loc[tabla["clave"] == "xgboost", "modelo"].item() == "XGBoost"
    assert tabla.loc[tabla["es_original"], "modelo"].item() == "DES-CPL"
    assert set(tabla["familia"]) == {"Modelo base", "Ensamble estático", "Modelo original", "Estado del arte"}


def test_tabla_sin_benchmarks_sota_sigue_funcionando():
    tabla = A.tabla_igual_presupuesto(_evaluacion(), None)
    assert "lightgbm" not in set(tabla["clave"]) and len(tabla) == 4


def test_lectura_dice_que_empata_cuando_esta_en_la_franja():
    tabla = A.tabla_igual_presupuesto(_evaluacion(), _sota())
    lineas = A.lectura_lideres_presupuesto(tabla)
    assert any("XGBoost" in l for l in lineas)
    assert any("empata" in l for l in lineas)  # 0,001 de diferencia < 0,005


def test_lectura_no_dice_que_empata_cuando_queda_lejos():
    evaluacion = _evaluacion()
    evaluacion.loc[evaluacion["método"] == "DES-CPL", "auc_pr"] -= 0.03
    tabla = A.tabla_igual_presupuesto(evaluacion, _sota())
    lineas = A.lectura_lideres_presupuesto(tabla)
    assert not any("empata" in l for l in lineas)
    assert any("fuera de la franja" in l for l in lineas)


@pytest.fixture
def sin_resultados(tmp_path, monkeypatch):
    """results/ vacío: simula un despliegue al que le faltan los CSV."""
    monkeypatch.setattr(C, "RESULTADOS", tmp_path)
    for funcion in (D.evaluacion_60, D.benchmarks_sota, D.conformal_protocolo):
        funcion.cache_clear()
    yield
    for funcion in (D.evaluacion_60, D.benchmarks_sota, D.conformal_protocolo):
        funcion.cache_clear()


def test_csv_ausentes_devuelven_none_y_no_fallan(sin_resultados):
    assert D.evaluacion_60() is None
    assert D.benchmarks_sota() is None
    assert D.conformal_protocolo() is None
    assert A.lideres_igual_presupuesto() is None


def test_falta_solo_sota_la_tabla_sale_sin_lightgbm_ni_catboost(sin_resultados, tmp_path):
    _evaluacion().to_csv(tmp_path / "des_cpl_evaluacion_60.csv", index=False)
    D.evaluacion_60.cache_clear()
    tabla = A.lideres_igual_presupuesto()
    assert tabla is not None and "catboost" not in set(tabla["clave"])


# --- Cifras de control con los CSV reales ---------------------------------- #
HAY_RESULTADOS = (C.RESULTADOS / "des_cpl_evaluacion_60.csv").exists() and (C.RESULTADOS / "benchmarks_sota.csv").exists()


@pytest.mark.skipif(not HAY_RESULTADOS, reason="faltan results/des_cpl_evaluacion_60.csv o benchmarks_sota.csv")
def test_cifras_de_control_a_igual_presupuesto():
    tabla = A.lideres_igual_presupuesto().set_index("clave")
    control = {"xgboost": 0.2290, "Pesos globales (Brier)": 0.2286, "catboost": 0.2285,
               "Apilamiento logístico": 0.2280, C.MODELO_ORIGINAL: 0.2279, "lightgbm": 0.2253}
    for clave, valor in control.items():
        assert round(tabla.loc[clave, "auc_pr_media"], 4) == valor, clave
    assert len(tabla) == 13  # 7 base + 2 estado del arte + 3 ensambles + DES-CPL
    assert tabla.index[0] == "xgboost"
