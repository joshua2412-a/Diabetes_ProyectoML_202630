"""Pruebas de la página /modelo-original: diferencias e intervalos, CSV ausentes,
las 13 comprobaciones, la ilustración y el explicador de DES-CPL.

Lo que usa datos sintéticos corre en cualquier entorno; lo que necesita el
artefacto o los CSV reales (``des_cpl.joblib``, ``results/``) se omite si no
están, porque en Render y en la integración continua solo hay lo que está en Git.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import des_cpl as dc
from src.estadistica import prueba_nadeau_bengio
from src.tablero import config as C
from src.tablero import datos as D
from src.tablero import evidencia as E
from src.tablero import explicador as X
from src.tablero.ilustracion import svg_comite

ORIGINAL = C.MODELO_ORIGINAL


def _matriz(sesgo=0.0, ruido=0.003, folds=5, semilla=0):
    """AUC-PR por fold de DES-CPL y de tres benchmarks (uno igual, uno peor, uno mejor)."""
    rng = np.random.default_rng(semilla)
    base = 0.22 + 0.01 * rng.standard_normal(folds)
    return pd.DataFrame({
        ORIGINAL: base,
        "xgboost": base + sesgo + ruido * rng.standard_normal(folds),
        "bayes": base - 0.03 + ruido * rng.standard_normal(folds),
        "lightgbm": base + 0.03 + ruido * rng.standard_normal(folds)})


# --- Diferencias e intervalos ---------------------------------------------- #
def test_intervalo_usa_varianza_de_nadeau_bengio_y_t_con_k_menos_1():
    matriz = _matriz()
    tabla = E.diferencias_con_intervalo(matriz, benchmarks=["xgboost"]).iloc[0]
    d = (matriz[ORIGINAL] - matriz["xgboost"]).to_numpy()
    mitad = stats.t.ppf(0.975, 4) * np.sqrt((1 / 5 + 1 / 4) * d.var(ddof=1))
    assert tabla["diferencia"] == pytest.approx(d.mean())
    assert tabla["ic_bajo"] == pytest.approx(d.mean() - mitad)
    assert tabla["ic_alto"] == pytest.approx(d.mean() + mitad)
    assert tabla["p"] == pytest.approx(prueba_nadeau_bengio(d)["p"])


def test_holm_se_aplica_sobre_todos_los_benchmarks_y_no_baja_el_p():
    tabla = E.diferencias_con_intervalo(_matriz(), benchmarks=["xgboost", "bayes", "lightgbm"])
    assert (tabla["p_holm"] >= tabla["p"] - 1e-12).all() and (tabla["p_holm"] <= 1).all()
    assert len(tabla) == 3


def test_categorias_frente_a_la_franja_de_equivalencia():
    tabla = E.diferencias_con_intervalo(_matriz(ruido=0.0005), benchmarks=["xgboost", "bayes", "lightgbm"])
    categoria = tabla.set_index("clave")["categoria"]
    assert categoria["bayes"] == E.MEJOR        # DES-CPL +0,03: el intervalo excluye el 0
    assert categoria["lightgbm"] == E.PEOR      # DES-CPL −0,03
    assert categoria["xgboost"] == E.EQUIVALENTE  # diferencia ~0 con poco ruido: cae dentro de ±0,005


def test_intervalo_ancho_cruza_el_cero_y_queda_sin_conclusion():
    tabla = E.diferencias_con_intervalo(_matriz(ruido=0.02), benchmarks=["xgboost"])
    assert tabla.iloc[0]["categoria"] == E.SIN_CONCLUSION


def test_ignora_benchmarks_que_no_estan_en_la_matriz():
    tabla = E.diferencias_con_intervalo(_matriz(), benchmarks=["xgboost", "catboost"])
    assert list(tabla["clave"]) == ["xgboost"]


HAY_60 = (C.RESULTADOS / "des_cpl_evaluacion_60.csv").exists() and (C.RESULTADOS / "benchmarks_sota.csv").exists()


@pytest.mark.skipif(not HAY_60, reason="faltan los CSV de la evaluación a 60 evaluaciones")
def test_control_frente_a_xgboost_con_los_csv_reales():
    matriz = E.matriz_por_fold(D.evaluacion_60(), D.benchmarks_sota())
    tabla = E.diferencias_con_intervalo(matriz).set_index("clave")
    assert len(tabla) == 9
    assert round(tabla.loc["xgboost", "diferencia"], 4) == -0.0010
    assert round(tabla.loc["xgboost", "p"], 2) == 0.60
    assert tabla.loc["xgboost", "categoria"] == E.SIN_CONCLUSION


# --- Regla de superioridad ------------------------------------------------- #
def test_las_cuatro_condiciones_no_se_cumplen_con_un_modelo_peor():
    matriz = _matriz(sesgo=0.01)      # el mejor benchmark supera a DES-CPL
    condiciones = E.condiciones_superioridad(matriz, None, None)
    assert [c["cumple"] for c in condiciones] == [False, None, False, None]
    assert all(c["valor"] for c in condiciones)


def test_condiciones_sin_datos_no_fallan():
    condiciones = E.condiciones_superioridad(None, None, None)
    assert len(condiciones) == 4 and all(c["cumple"] is None for c in condiciones)


@pytest.mark.skipif(not HAY_60, reason="faltan los CSV de la evaluación a 60 evaluaciones")
def test_ninguna_condicion_se_cumple_con_los_csv_reales():
    matriz = E.matriz_por_fold(D.evaluacion_60(), D.benchmarks_sota())
    adicionales = D.leer_csv_opcional(C.RESULTADOS / "adicionales" / "evaluacion_conjuntos.csv")
    if adicionales is None:
        pytest.skip("falta results/adicionales/evaluacion_conjuntos.csv")
    condiciones = E.condiciones_superioridad(matriz, E.pruebas_protocolo(), adicionales)
    assert [c["cumple"] for c in condiciones] == [False, False, False, False]


# --- Pasos 4 y 5 ------------------------------------------------------------ #
def _evaluacion_con_variantes():
    filas = []
    for fold in range(5):
        filas.append({"fold": fold, "método": "DES-CPL", "familia": "Modelo original", "auc_pr": 0.22 + 0.01 * fold,
                      "lambda": [1, 1, 1, 0.5, 0.9][fold]})
        filas.append({"fold": fold, "método": "Ablación: sin poda (7 modelos)", "familia": "Ablación",
                      "auc_pr": 0.22 + 0.01 * fold + 0.001, "lambda": np.nan})
        filas.append({"fold": fold, "método": "DES-CPL inicial (sin poda ni encogimiento)", "familia": "Versión inicial",
                      "auc_pr": 0.22 + 0.01 * fold - 0.004, "lambda": np.nan})
        filas.append({"fold": fold, "método": "xgboost", "familia": "Modelo base", "auc_pr": 0.2, "lambda": np.nan})
    return pd.DataFrame(filas)


def test_lambda_por_fold_lee_el_de_cada_fold():
    assert E.lambda_por_fold(_evaluacion_con_variantes()).tolist() == [1, 1, 1, 0.5, 0.9]


def test_ablacion_es_variante_menos_completo_y_solo_de_las_dos_familias():
    tabla = E.ablacion(_evaluacion_con_variantes()).set_index("variante")
    assert set(tabla.index) == {"Ablación: sin poda (7 modelos)", "DES-CPL inicial (sin poda ni encogimiento)"}
    assert tabla.loc["Ablación: sin poda (7 modelos)", "diferencia"] == pytest.approx(0.001)
    assert tabla.loc["DES-CPL inicial (sin poda ni encogimiento)", "diferencia"] == pytest.approx(-0.004)
    assert tabla["sd"].max() < 1e-12


def test_rangos_en_conjuntos_cuenta_el_principal_y_los_adicionales():
    evaluacion = pd.DataFrame([{"fold": f, "método": m, "familia": "Modelo base" if m != "DES-CPL" else "Modelo original",
                                "auc_pr": v, "auc_roc": 0.6, "brier": 0.1}
                               for f in range(2) for m, v in [("xgboost", 0.30), ("DES-CPL", 0.25)]])
    adicionales = pd.DataFrame([{"conjunto": c, "fold": f, "método": m, "auc_pr": v,
                                 "familia": "Modelo base" if m != "DES-CPL" else "Modelo original"}
                                for c in ("credito", "banco") for f in range(2)
                                for m, v in [("xgboost", 0.2), ("DES-CPL", 0.4)]])
    rangos = E.rangos_en_conjuntos(evaluacion, None, adicionales).set_index("clave")
    assert rangos.loc[ORIGINAL, "rango_4"] == pytest.approx((2 + 1 + 1) / 3)   # 3 conjuntos en total
    assert rangos.loc[ORIGINAL, "rango_3"] == 1.0 and rangos.loc["xgboost", "rango_3"] == 2.0


def test_robustez_perdidas_y_razon_con_des_cpl():
    tabla = pd.DataFrame({"método": ["DES-CPL", "xgboost"], "original": [0.23, 0.23],
                          "ruido_05": [0.22, 0.21], "tamano_25": [0.21, 0.19]})
    largo = E.robustez_perdidas(tabla)
    x = largo[(largo["método"] == "xgboost") & (largo["escenario"] == "tamano_25")].iloc[0]
    assert x["perdida"] == pytest.approx(0.04)
    assert x["razon_con_des_cpl"] == pytest.approx(2.0)


# --- CSV ausentes ------------------------------------------------------------ #
@pytest.fixture
def sin_resultados(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RESULTADOS", tmp_path)
    monkeypatch.setattr(C, "ORIGINAL", tmp_path / "no_hay")
    yield tmp_path


def test_csv_ausentes_devuelven_none_y_no_fallan(sin_resultados):
    assert E.pruebas_protocolo() is None
    assert E.diagnostico_r2() is None
    assert E.semillas_resumen() is None
    assert E.sintesis_comprobaciones() == {"verificaciones": None, "exploraciones": None}


def test_sintesis_lee_los_csv_cuando_existen(sin_resultados):
    (sin_resultados / "verificaciones").mkdir()
    pd.DataFrame({"verificación": ["V1", "V3"], "se mantiene la conclusión": ["Sí", "No"]}).to_csv(
        sin_resultados / "verificaciones" / "sintesis.csv", index=False)
    s = E.sintesis_comprobaciones()
    assert s["verificaciones"] == {"total": 2, "mantienen": 1, "no_mantienen": ["V3"]}
    assert s["exploraciones"] is None


def test_diagnostico_r2_toma_el_maximo_de_los_r2(sin_resultados):
    pd.DataFrame({"espacio": ["perfil", "perfil + variables", "perfil"],
                  "análisis": ["R² de pérdida a − b", "R² de pérdida c − d", "Competencia aprendida"],
                  "tau": [np.nan, np.nan, 0.01], "valor_seleccion": [0.004, 0.0125, 0.9],
                  "auc_pr_evaluacion": [np.nan, np.nan, 0.25]}).to_csv(
        sin_resultados / "des_cpl_diagnostico_errores.csv", index=False)
    r2 = E.diagnostico_r2()
    assert r2["maximo"] == pytest.approx(0.0125) and r2["n_pares"] == 2


# --- Las 13 comprobaciones --------------------------------------------------- #
def test_trece_comprobaciones_ninguna_cambia_el_veredicto():
    claves = [c[0] for c in C.COMPROBACIONES]
    assert claves == [f"V{i}" for i in range(1, 10)] + [f"E{i}" for i in range(1, 5)]
    assert all(c[3] == "No" for c in C.COMPROBACIONES)
    assert all(c[1] and c[2] for c in C.COMPROBACIONES)


def test_legible_cambia_coma_decimal_y_nombres_de_modelo():
    assert E.legible("rango DES-CPL 2.8 vs xgboost 1.8") == "rango DES-CPL 2,8 vs XGBoost 1,8"


# --- Ilustración -------------------------------------------------------------- #
def test_ilustracion_es_un_svg_valido_con_los_cuatro_modelos():
    svg = svg_comite()
    raiz = ET.fromstring(svg)                       # XML bien formado
    assert raiz.tag.endswith("svg") and "\n" not in svg
    for nombre in ("Árbol", "Random Forest", "XGBoost", "SVM"):
        assert nombre in svg
    assert "NaN" not in svg and "style=\"--i:" in svg


# --- Explicador: matemática de la combinación (sin artefacto) ---------------- #
def _datos(c_local=(0.0931, 0.0930, 0.0926, 0.0933), c_global=(0.0934, 0.0932, 0.0928, 0.0935),
           p=(0.05, 0.07, 0.09, 0.06)):
    return {"competencia_local": list(c_local), "competencia_global": list(c_global), "probabilidades": list(p)}


def test_combinacion_mezcla_pesos_y_probabilidad_a_mano():
    datos = _datos()
    r = X.combinacion(datos, 0.9, 0.0003)
    c = 0.1 * np.array(datos["competencia_local"]) + 0.9 * np.array(datos["competencia_global"])
    w = np.exp(-(c - c.min()) / 0.0003)
    w /= w.sum()
    assert r["pesos"] == pytest.approx(w.tolist())
    assert r["p"] == pytest.approx(float(w @ np.array(datos["probabilidades"])))
    assert sum(r["pesos"]) == pytest.approx(1.0)


def test_lambda_1_es_pesos_globales_y_lambda_0_es_local():
    datos = _datos()
    assert X.combinacion(datos, 1.0, 0.0003)["pesos"] == pytest.approx(
        dc.pesos_softmax(np.array([datos["competencia_global"]]), 0.0003)[0].tolist())
    assert X.combinacion(datos, 0.0, 0.0003)["pesos"] == pytest.approx(
        dc.pesos_softmax(np.array([datos["competencia_local"]]), 0.0003)[0].tolist())


def test_tau_grande_es_el_promedio_simple_y_lambda_fuera_de_rango_falla():
    datos = _datos()
    assert X.combinacion(datos, 0.9, 1e9)["p"] == pytest.approx(np.mean(datos["probabilidades"]), rel=1e-6)
    with pytest.raises(ValueError):
        X.combinacion(datos, 1.5, 0.0003)


def test_tau_del_artefacto_esta_en_el_deslizador():
    assert 0.0003 in C.TAUS and C.TAUS == tuple(sorted(C.TAUS))


# --- Explicador con el artefacto real ---------------------------------------- #
HAY_ARTEFACTO = (C.ARTEFACTOS / "des_cpl.joblib").exists() and (C.PRUEBA).exists()


@pytest.mark.skipif(not HAY_ARTEFACTO, reason="falta artifacts/des_cpl.joblib o los datos de prueba")
@pytest.mark.parametrize("indice", [0, 5, 123, 19000])
def test_explicador_con_lambda_09_y_tau_00003_reproduce_la_probabilidad_del_artefacto(indice):
    parametros = X.parametros_artefacto()
    assert (parametros["lam"], parametros["tau"], parametros["k"]) == (0.9, 0.0003, 20)
    r = X.combinacion(X.caso(indice), parametros["lam"], parametros["tau"])
    assert r["p"] == pytest.approx(X.p_artefacto(indice), abs=1e-12)
    assert sum(r["pesos"]) == pytest.approx(1.0)


@pytest.mark.skipif(not HAY_ARTEFACTO, reason="falta artifacts/des_cpl.joblib o los datos de prueba")
def test_caso_trae_20_vecinos_por_clase_y_es_serializable():
    import json

    datos = X.caso(5)
    json.dumps(datos)                                   # apto para un dcc.Store
    assert len(datos["vecinos_y"]) == 40 and sum(datos["vecinos_y"]) == 20
    assert len(datos["modelos"]) == len(datos["probabilidades"]) == len(datos["competencia_local"]) == 4
    proyeccion = X.proyeccion()
    assert len(proyeccion["fondo"]) == C.PUNTOS_FONDO_PROYECCION
    assert 0.9 < sum(proyeccion["varianza"]) <= 1.0     # dos componentes recogen casi todo


@pytest.mark.skipif(not HAY_ARTEFACTO, reason="falta artifacts/des_cpl.joblib o los datos de prueba")
def test_pacientes_al_azar_son_reproducibles_y_los_readmitidos_lo_son():
    assert X.indice_al_azar(7) == X.indice_al_azar(7)
    i = X.indice_al_azar(7, solo_readmitidos=True)
    assert D.prueba().iloc[i]["readmitted_30"] == 1
