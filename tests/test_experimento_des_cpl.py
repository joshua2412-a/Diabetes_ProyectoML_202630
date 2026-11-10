"""Pruebas del protocolo experimental de DES-CPL (src/experimento_des_cpl.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from src import experimento_des_cpl as exd

MODELOS = ("logistica", "bayes", "arbol")


@pytest.fixture
def mejores():
    parametros = {
        "logistica": {"modelo__C": 1.0, "modelo__penalty": "l2"},
        "bayes": {"modelo__var_smoothing": 0.01},
        "arbol": {"modelo__max_depth": 3, "modelo__min_samples_leaf": 5,
                  "modelo__criterion": "gini"},
    }
    maestra = pd.DataFrame([{
        "modelo": m, "balanceo": "ninguno", "optimizador": "grid",
        "estado": "completada", "auc_pr_media": 0.3,
        "auc_pr_por_fold": json.dumps([0.3] * 5),
        "hiperparametros_por_fold": json.dumps([p] * 5)}
        for m, p in parametros.items()])
    return exd.mejores_combinaciones(maestra)


@pytest.fixture
def fold_calculado(datos, roles, mejores, tmp_path):
    X, y, grupos = datos
    particiones = exd.particiones_externas(X, y, grupos)
    seleccion, evaluado = exd.ejecutar_fold(0, particiones[0], X, y, grupos,
                                            roles, mejores, tmp_path,
                                            verbose=False)
    return X, y, grupos, particiones, seleccion, evaluado, tmp_path


def test_mejores_combinaciones_en_orden_fijo(mejores):
    assert list(mejores.index) == ["logistica", "bayes", "arbol"]
    assert mejores.loc["bayes", "hiperparametros_por_fold"][2] == {
        "modelo__var_smoothing": 0.01}


def test_particiones_externas_sin_pacientes_compartidos(datos):
    X, y, grupos = datos
    for entrenamiento, evaluacion in exd.particiones_externas(X, y, grupos):
        assert not set(grupos.iloc[entrenamiento]) & set(grupos.iloc[evaluacion])


def test_fold_produce_predicciones_completas(fold_calculado):
    _, _, _, particiones, seleccion, evaluado, _ = fold_calculado
    entrenamiento, evaluacion = particiones[0]
    assert len(seleccion) == len(entrenamiento)
    assert len(evaluado) == len(evaluacion)
    for m in MODELOS:
        assert seleccion[f"oof_{m}"].between(0, 1).all()
        assert evaluado[f"final_{m}"].between(0, 1).all()
        assert evaluado[f"pliegue_{m}"].between(0, 1).all()


def test_modelo_final_reproduce_el_ajuste_del_e2(fold_calculado, roles, mejores):
    """El modelo final es el pipeline del E2 ajustado con todo el entrenamiento externo."""
    from src.balanceo import construir_pipeline
    X, y, _, particiones, _, evaluado, _ = fold_calculado
    entrenamiento, evaluacion = particiones[0]
    pipeline = construir_pipeline("logistica", "ninguno", roles).set_params(
        **mejores.loc["logistica", "hiperparametros_por_fold"][0])
    esperado = pipeline.fit(X.iloc[entrenamiento], y.iloc[entrenamiento]) \
        .predict_proba(X.iloc[evaluacion])[:, 1]
    np.testing.assert_allclose(evaluado["final_logistica"], esperado, rtol=1e-6)


def test_reanuda_sin_recalcular(fold_calculado, roles, mejores, capsys):
    X, y, grupos, particiones, seleccion, _, carpeta = fold_calculado
    antes = (carpeta / "seleccion_fold0.csv").stat().st_mtime_ns
    exd.ejecutar_fold(0, particiones[0], X, y, grupos, roles, mejores, carpeta)
    assert "[omitido]" in capsys.readouterr().out
    assert (carpeta / "seleccion_fold0.csv").stat().st_mtime_ns == antes
    estado = exd.progreso(carpeta, folds=[0, 1], modelos=MODELOS)
    assert list(estado["estado"]) == ["Completo", "Sin empezar"]


def test_reanuda_un_modelo_faltante(datos, roles, mejores, tmp_path):
    X, y, grupos = datos
    particion = exd.particiones_externas(X, y, grupos)[0]
    exd.ejecutar_fold(0, particion, X, y, grupos, roles, mejores, tmp_path,
                      modelos=["logistica"], verbose=False)
    assert exd.progreso(tmp_path, [0], MODELOS)["estado"].iloc[0] == "En curso"
    sel, _ = exd.ejecutar_fold(0, particion, X, y, grupos, roles, mejores,
                               tmp_path, verbose=False)
    assert {f"oof_{m}" for m in MODELOS} <= set(sel.columns)


def test_evaluacion_incluye_todos_los_metodos(fold_calculado):
    *_, seleccion, evaluado, _ = fold_calculado
    tabla = exd.evaluar_fold(seleccion, evaluado, MODELOS, ks=(3, 5),
                             taus=(0.0, 0.01, np.inf), lambdas=(0.0, 1.0), podas=(2, None))
    metodos = set(tabla["método"])
    assert set(MODELOS) | {"Promedio simple", "Apilamiento logístico",
                           "Pesos globales (Brier)"} <= metodos
    assert set(exd.VARIANTES) <= metodos
    assert tabla[["auc_pr", "auc_roc", "brier"]].notna().all().all()
    dura = tabla.set_index("método").loc["Ablación: selección dura (τ = 0)"]
    assert dura["tau"] == 0.0


def test_temperatura_infinita_reproduce_el_promedio(fold_calculado):
    *_, seleccion, evaluado, _ = fold_calculado
    tabla = exd.evaluar_fold(seleccion, evaluado, MODELOS, ks=(3,),
                             taus=(np.inf,)).set_index("método")
    assert tabla.loc["DES-CPL", "auc_pr"] == pytest.approx(
        tabla.loc["Promedio simple", "auc_pr"])


def test_evaluar_guarda_y_no_repite(fold_calculado, tmp_path):
    *_, carpeta = fold_calculado
    ruta = tmp_path / "evaluacion.csv"
    primera = exd.evaluar(carpeta, [0, 1], ruta, MODELOS, ks=(3,), taus=(0.01,))
    assert set(primera["fold"]) == {0}
    segunda = exd.evaluar(carpeta, [0, 1], ruta, MODELOS, ks=(3,), taus=(0.01,))
    assert len(segunda) == len(primera)


def test_poda_y_encogimiento_se_eligen_en_seleccion(fold_calculado):
    *_, seleccion, evaluado, _ = fold_calculado
    tabla = exd.evaluar_fold(seleccion, evaluado, MODELOS, ks=(3,),
                             taus=(0.01,), lambdas=(0.0, 0.5, 1.0),
                             podas=(2, None)).set_index("método")
    assert tabla.loc["DES-CPL", "lambda"] in (0.0, 0.5, 1.0)
    inicial = tabla.loc["DES-CPL inicial (sin poda ni encogimiento)"]
    assert inicial["lambda"] == 0.0
    assert inicial["modelos_combinados"] == ", ".join(MODELOS)
    # La búsqueda completa incluye a la inicial: en selección no puede ser peor.
    assert tabla.loc["DES-CPL", "auc_pr_seleccion"] >= inicial["auc_pr_seleccion"] - 1e-12


def test_con_presupuesto_elige_la_mejor_interna_entre_las_primeras(mejores, tmp_path):
    """Con presupuesto 3 se elige el mejor puntaje interno de los índices 0-2."""
    filas_ev, filas_re = [], []
    for modelo in mejores.index:
        for fold in range(1, 6):
            for indice, puntaje in enumerate([0.1, 0.3, 0.2, 0.9]):   # el 3 queda fuera
                parametros = json.dumps({"modelo__marca": indice})
                filas_ev.append({"modelo": modelo, "balanceo": "ninguno", "fold": fold,
                                 "indice": indice, "parametros": parametros,
                                 "puntaje_interno": puntaje})
                filas_re.append({"modelo": modelo, "balanceo": "ninguno", "fold": fold,
                                 "indice": indice, "parametros": parametros,
                                 "auc_pr": 0.20 + indice / 100})
    pd.DataFrame(filas_ev).to_csv(tmp_path / "ev.csv", index=False)
    pd.DataFrame(filas_re).to_csv(tmp_path / "re.csv", index=False)
    tabla = exd.con_presupuesto(mejores, tmp_path / "ev.csv", tmp_path / "re.csv", 3)
    assert tabla.loc["bayes", "hiperparametros_por_fold"][0] == {"modelo__marca": 1}
    assert tabla.loc["bayes", "auc_pr_por_fold"] == pytest.approx([0.21] * 5)
    with pytest.raises(ValueError):
        exd.con_presupuesto(mejores, tmp_path / "ev.csv", tmp_path / "re.csv", 10)


def test_devuelve_predicciones_alineadas(fold_calculado):
    *_, seleccion, evaluado, _ = fold_calculado
    tabla, pred = exd.evaluar_fold(seleccion, evaluado, MODELOS, ks=(3,), taus=(0.01,),
                                   lambdas=(0.0, 1.0), podas=(None,),
                                   devolver_predicciones=True)
    assert set(tabla["método"]) <= set(pred)
    assert len(pred["DES-CPL"]) == len(evaluado)
    assert len(pred["DES-CPL (fuera de pliegue)"]) == len(seleccion)
    from sklearn.metrics import average_precision_score
    fila = tabla.set_index("método").loc["DES-CPL"]
    assert average_precision_score(evaluado["y"], pred["DES-CPL"]) == pytest.approx(fila["auc_pr"])
