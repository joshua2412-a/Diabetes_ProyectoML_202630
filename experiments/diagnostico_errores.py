"""
¿Se puede predecir qué modelo base se equivoca menos en cada paciente?

Diagnóstico del piloto de DES-CPL (capítulo 13). DES-CPL supone que la
calidad relativa de los modelos cambia de una región a otra y que esa
diferencia se puede estimar. Este script lo contrasta directamente en el
fold 0: aprende, con un modelo flexible, la diferencia de pérdida de Brier
entre pares de modelos a partir de

* el perfil de salida (logit de las siete probabilidades), que es el
  espacio que usa DES-CPL, y
* el perfil más las 38 variables originales del paciente,

y mide cuánto de esa diferencia se predice fuera de pliegue (R², pliegues
agrupados por paciente). Si el R² es cercano a cero en los dos espacios, no
hay información con qué elegir modelo por paciente, y ninguna variante de
selección dinámica puede superar a los pesos globales.

También evalúa la variante «competencia aprendida»: los pesos softmax con la
pérdida predicha por ese modelo en lugar de la de los vecinos.

Uso (raíz del repositorio, después del piloto):

    python experiments/diagnostico_errores.py

Escribe results/des_cpl_diagnostico_errores.csv. Tarda unos minutos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import configuracion as cfg                        # noqa: E402
from src.des_cpl import combinar, perfil, pesos_softmax    # noqa: E402
from src.experimento_des_cpl import MODELOS                 # noqa: E402

RESULTADOS = RAIZ / "results"
PARES = [("random_forest", "xgboost"), ("random_forest", "logistica"),
         ("random_forest", "arbol"), ("xgboost", "svm"), ("logistica", "knn")]
TAUS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def _regresor():
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, min_samples_leaf=300,
        l2_regularization=1.0, categorical_features="from_dtype",
        random_state=cfg.SEMILLA)


def _espacio(nombre, perfiles, X):
    if nombre == "perfil":
        return pd.DataFrame(perfiles, columns=[f"logit_{m}" for m in MODELOS])
    return pd.concat([_espacio("perfil", perfiles, X),
                      X.reset_index(drop=True)], axis=1)


def diagnosticar(seleccion, evaluado, X_sel, X_ev, pliegues=3):
    """Tabla de R² por par y espacio, y AUC-PR de la competencia aprendida."""
    P_sel = seleccion[[f"oof_{m}" for m in MODELOS]].to_numpy()
    P_ev = evaluado[[f"final_{m}" for m in MODELOS]].to_numpy()
    y_sel, y_ev = seleccion["y"].to_numpy(), evaluado["y"].to_numpy()
    grupos = seleccion["grupo"].to_numpy()
    perfil_sel = perfil(P_sel)
    perfil_ev = perfil(evaluado[[f"pliegue_{m}" for m in MODELOS]].to_numpy())
    perdida = (P_sel - y_sel[:, None]) ** 2
    particiones = list(GroupKFold(pliegues).split(P_sel, y_sel, grupos))
    filas = []

    for espacio in ("perfil", "perfil + variables"):
        Z_sel = _espacio(espacio, perfil_sel, X_sel)
        Z_ev = _espacio(espacio, perfil_ev, X_ev)
        for a, b in PARES:
            i, j = MODELOS.index(a), MODELOS.index(b)
            d = perdida[:, i] - perdida[:, j]
            pred = np.zeros_like(d)
            for ajuste, valida in particiones:
                pred[valida] = _regresor().fit(Z_sel.iloc[ajuste], d[ajuste]).predict(Z_sel.iloc[valida])
            r2 = 1 - ((d - pred) ** 2).sum() / ((d - d.mean()) ** 2).sum()
            filas.append({"espacio": espacio, "análisis": f"R² de pérdida {a} − {b}",
                          "tau": np.nan, "valor_seleccion": r2, "auc_pr_evaluacion": np.nan})
            print(f"[{espacio}] R² fuera de pliegue, {a} − {b}: {r2:.4f}")

        # Competencia aprendida: pérdida predicha de cada modelo.
        pred_sel = np.zeros_like(perdida)
        pred_ev = np.zeros((len(y_ev), len(MODELOS)))
        for m in range(len(MODELOS)):
            for ajuste, valida in particiones:
                pred_sel[valida, m] = _regresor().fit(
                    Z_sel.iloc[ajuste], perdida[ajuste, m]).predict(Z_sel.iloc[valida])
            pred_ev[:, m] = _regresor().fit(Z_sel, perdida[:, m]).predict(Z_ev)
        for tau in TAUS:
            sel = average_precision_score(y_sel, combinar(pesos_softmax(pred_sel, tau), P_sel))
            ev = average_precision_score(y_ev, combinar(pesos_softmax(pred_ev, tau), P_ev))
            filas.append({"espacio": espacio, "análisis": "Competencia aprendida",
                          "tau": tau, "valor_seleccion": sel, "auc_pr_evaluacion": ev})
            print(f"[{espacio}] competencia aprendida, tau={tau:g}: "
                  f"selección {sel:.4f} | evaluación {ev:.4f}")
    return pd.DataFrame(filas)


def main():
    roles = json.loads((RAIZ / "data" / "processed" / "roles_variables.json")
                       .read_text(encoding="utf-8"))
    predictores = (roles["numericas"] + roles["ordinales"] + roles["binarias"]
                   + roles["categoricas"])
    train = pd.read_parquet(RAIZ / "data" / "processed" / "diabetes_train.parquet")
    X = train[predictores].copy()
    for columna in roles["categoricas"]:
        X[columna] = X[columna].astype("category")
    seleccion = pd.read_csv(RESULTADOS / "des_cpl" / "seleccion_fold0.csv")
    evaluado = pd.read_csv(RESULTADOS / "des_cpl" / "evaluacion_fold0.csv")
    tabla = diagnosticar(seleccion, evaluado, X.iloc[seleccion["fila"]],
                         X.iloc[evaluado["fila"]])
    tabla.to_csv(RESULTADOS / "des_cpl_diagnostico_errores.csv", index=False)
    print("\nReferencia del fold 0: Random Forest 0,2535; pesos globales 0,2540.")


if __name__ == "__main__":
    main()
