"""
Cribado rápido de candidatos a modelo original (fold 0).

Antes de implementar desde cero otro modelo original, se mide si su familia
tiene alguna posibilidad de superar al mejor modelo base. Cada candidato de
la lista de modelos no usados en diabetes se prueba con una implementación
de biblioteca (o una versión mínima propia si no existe), con
hiperparámetros razonables y sin búsqueda, en el mismo fold externo 0 que
el piloto de DES-CPL. Es una cota optimista del candidato: si ni la versión
de biblioteca se acerca a Random Forest (AUC-PR 0,2535 en ese fold), una
implementación propia tampoco lo hará.

Para cada candidato se reporta el AUC-PR y el AUC-ROC en el fold de
evaluación, el tiempo y un intervalo bootstrap del 95 % de la diferencia de
AUC-PR frente a Random Forest, remuestreando pacientes (500 réplicas).

Uso (raíz del repositorio, después del piloto de DES-CPL):

    python experiments/cribado_modelos.py
    python experiments/cribado_modelos.py --folds 0 1 2 3 4 --candidatos "Extra Trees"

La segunda forma repite los candidatos indicados en los cinco folds externos
(necesita los cinco folds de DES-CPL) y guarda results/cribado_modelos_folds.csv.

Guarda cada candidato al terminarlo en results/cribado_modelos_fold0.csv;
si se interrumpe, continúa con los que faltan.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeClassifier

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import configuracion as cfg                          # noqa: E402
from src.preprocesamiento import construir_preprocesador      # noqa: E402

SEMILLA = cfg.SEMILLA
RESULTADOS = RAIZ / "results"
RUTA = RESULTADOS / "cribado_modelos_fold0.csv"
RUTA_FOLDS = RESULTADOS / "cribado_modelos_folds.csv"


class RotationForest(ClassifierMixin, BaseEstimator):
    """Rotation Forest mínimo (Rodríguez, Kuncheva y Alonso, 2006).

    Cada árbol se entrena sobre los datos rotados: las variables se parten
    al azar en ``k`` subconjuntos y cada uno se rota con un PCA ajustado en
    una submuestra bootstrap del 75 %.
    """

    def __init__(self, n_estimators=50, n_subconjuntos=6, max_depth=12,
                 min_samples_leaf=20, random_state=SEMILLA):
        self.n_estimators = n_estimators
        self.n_subconjuntos = n_subconjuntos
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _rotar(self, X, rotacion):
        return np.hstack([pca.transform(X[:, cols]) for cols, pca in rotacion])

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        aleatorio = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        self.miembros_ = []
        for _ in range(self.n_estimators):
            orden = aleatorio.permutation(X.shape[1])
            rotacion = []
            for cols in np.array_split(orden, self.n_subconjuntos):
                filas = aleatorio.choice(len(X), int(0.75 * len(X)), replace=True)
                rotacion.append((cols, PCA(random_state=0).fit(X[np.ix_(filas, cols)])))
            arbol = DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                random_state=int(aleatorio.integers(1 << 31)))
            self.miembros_.append((rotacion, arbol.fit(self._rotar(X, rotacion), y)))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.mean([a.predict_proba(self._rotar(X, r)) for r, a in self.miembros_], axis=0)


def candidatos(roles):
    """Nombre -> (estimador sobre la matriz preprocesada, nota)."""
    lista = {
        "Extra Trees": (ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=10, max_features=0.3,
            n_jobs=cfg.N_JOBS, random_state=SEMILLA), "biblioteca"),
        "Rotation Forest": (RotationForest(), "implementación mínima propia"),
        "Random Subspace": (BaggingClassifier(
            DecisionTreeClassifier(min_samples_leaf=20, max_depth=14),
            n_estimators=300, max_features=0.5, bootstrap=False,
            n_jobs=cfg.N_JOBS, random_state=SEMILLA), "biblioteca"),
        "Boosting tipo LogitBoost (HistGB)": (HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
            min_samples_leaf=100, l2_regularization=1.0, early_stopping=True,
            random_state=SEMILLA), "biblioteca, referencia de boosting"),
        "Modelo aditivo (GAM con splines)": (make_pipeline(
            SplineTransformer(n_knots=6, degree=3),
            LogisticRegression(C=0.05, max_iter=3000)), "aproximación con splines"),
        "Regresión logística con kernel (Nyström)": (make_pipeline(
            Nystroem(kernel="rbf", gamma=0.01, n_components=500, random_state=SEMILLA),
            LogisticRegression(C=0.1, max_iter=3000)), "aproximación de Nyström"),
        "QDA": (QuadraticDiscriminantAnalysis(reg_param=0.5), "biblioteca"),
    }
    try:
        from imblearn.ensemble import RUSBoostClassifier
        lista["RUSBoost"] = (RUSBoostClassifier(
            DecisionTreeClassifier(max_depth=4), n_estimators=300,
            learning_rate=0.1, random_state=SEMILLA), "imbalanced-learn")
    except ImportError:
        pass
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
        lista["EBM"] = (ExplainableBoostingClassifier(random_state=SEMILLA), "interpret")
    except ImportError:
        pass
    return lista


def diferencia_bootstrap(y, p, p_ref, grupos, replicas=500, semilla=SEMILLA):
    """IC 95 % de AP(p) - AP(p_ref) remuestreando pacientes."""
    aleatorio = np.random.default_rng(semilla)
    pacientes, inversa = np.unique(grupos, return_inverse=True)
    filas_por_paciente = pd.Series(np.arange(len(y))).groupby(inversa).agg(list).to_list()
    difs = []
    for _ in range(replicas):
        elegidos = aleatorio.integers(0, len(pacientes), len(pacientes))
        idx = np.concatenate([filas_por_paciente[i] for i in elegidos])
        difs.append(average_precision_score(y[idx], p[idx])
                    - average_precision_score(y[idx], p_ref[idx]))
    return np.percentile(difs, [2.5, 97.5])


def folds(nombres, lista_folds):
    """Repite los candidatos indicados en varios folds externos."""
    cfg.N_JOBS = 4
    roles = json.loads((RAIZ / "data" / "processed" / "roles_variables.json")
                       .read_text(encoding="utf-8"))
    predictores = (roles["numericas"] + roles["ordinales"] + roles["binarias"]
                   + roles["categoricas"])
    train = pd.read_parquet(RAIZ / "data" / "processed" / "diabetes_train.parquet")
    X, y = train[predictores], train[roles["objetivo"]].to_numpy()
    todos = candidatos(roles)
    hechos = (pd.read_csv(RUTA_FOLDS) if RUTA_FOLDS.exists()
              else pd.DataFrame(columns=["candidato", "fold"]))
    for fold in lista_folds:
        seleccion = pd.read_csv(RESULTADOS / "des_cpl" / f"seleccion_fold{fold}.csv",
                                usecols=["fila"])["fila"].to_numpy()
        evaluado = pd.read_csv(RESULTADOS / "des_cpl" / f"evaluacion_fold{fold}.csv",
                               usecols=["fila", "y", "final_random_forest", "final_xgboost"])
        pendientes = [n for n in nombres
                      if not ((hechos["candidato"] == n) & (hechos["fold"] == fold)).any()]
        if not pendientes:
            continue
        prepro = construir_preprocesador(roles, salida_densa=True).fit(X.iloc[seleccion])
        X_tr, X_ev = prepro.transform(X.iloc[seleccion]), prepro.transform(X.iloc[evaluado["fila"]])
        y_ev = evaluado["y"].to_numpy()
        for nombre in pendientes:
            inicio = time.perf_counter()
            p = clone(todos[nombre][0]).fit(X_tr, y[seleccion]).predict_proba(X_ev)[:, 1]
            fila = {"candidato": nombre, "fold": fold,
                    "auc_pr": average_precision_score(y_ev, p),
                    "auc_pr_rf": average_precision_score(y_ev, evaluado["final_random_forest"]),
                    "auc_pr_xgb": average_precision_score(y_ev, evaluado["final_xgboost"]),
                    "segundos": time.perf_counter() - inicio}
            hechos = pd.concat([hechos, pd.DataFrame([fila])], ignore_index=True)
            hechos.to_csv(RUTA_FOLDS, index=False)
            print(f"[listo] fold {fold}, {nombre}: AUC-PR {fila['auc_pr']:.4f} | RF "
                  f"{fila['auc_pr_rf']:.4f} | XGB {fila['auc_pr_xgb']:.4f}")
    resumen = hechos.groupby("candidato")[["auc_pr", "auc_pr_rf", "auc_pr_xgb"]].agg(["mean", "std"])
    print("\n" + resumen.round(4).to_string())


def main():
    cfg.N_JOBS = 4
    roles = json.loads((RAIZ / "data" / "processed" / "roles_variables.json")
                       .read_text(encoding="utf-8"))
    predictores = (roles["numericas"] + roles["ordinales"] + roles["binarias"]
                   + roles["categoricas"])
    train = pd.read_parquet(RAIZ / "data" / "processed" / "diabetes_train.parquet")
    seleccion = pd.read_csv(RESULTADOS / "des_cpl" / "seleccion_fold0.csv",
                            usecols=["fila"])["fila"].to_numpy()
    evaluado = pd.read_csv(RESULTADOS / "des_cpl" / "evaluacion_fold0.csv",
                           usecols=["fila", "y", "grupo", "final_random_forest"])
    X, y = train[predictores], train[roles["objetivo"]].to_numpy()
    prepro = construir_preprocesador(roles, salida_densa=True).fit(X.iloc[seleccion])
    X_tr = prepro.transform(X.iloc[seleccion])
    X_ev = prepro.transform(X.iloc[evaluado["fila"]])
    y_tr, y_ev = y[seleccion], evaluado["y"].to_numpy()
    p_rf = evaluado["final_random_forest"].to_numpy()

    hechos = pd.read_csv(RUTA) if RUTA.exists() else pd.DataFrame(columns=["candidato"])
    print(f"Referencia: Random Forest del E2, AUC-PR {average_precision_score(y_ev, p_rf):.4f}\n")
    for nombre, (estimador, nota) in candidatos(roles).items():
        if nombre in set(hechos["candidato"]):
            print(f"[omitido] {nombre}")
            continue
        inicio = time.perf_counter()
        try:
            p = clone(estimador).fit(X_tr, y_tr).predict_proba(X_ev)[:, 1]
        except Exception as error:  # noqa: BLE001 - un candidato no detiene el cribado
            print(f"[error] {nombre}: {error}")
            continue
        segundos = time.perf_counter() - inicio
        bajo, alto = diferencia_bootstrap(y_ev, p, p_rf, evaluado["grupo"].to_numpy())
        fila = {"candidato": nombre, "implementacion": nota,
                "auc_pr": average_precision_score(y_ev, p),
                "auc_roc": roc_auc_score(y_ev, p),
                "dif_vs_rf": average_precision_score(y_ev, p) - average_precision_score(y_ev, p_rf),
                "ic95_bajo": bajo, "ic95_alto": alto, "segundos": segundos}
        hechos = pd.concat([hechos, pd.DataFrame([fila])], ignore_index=True)
        hechos.to_csv(RUTA, index=False)
        print(f"[listo] {nombre}: AUC-PR {fila['auc_pr']:.4f} ({fila['dif_vs_rf']:+.4f} vs RF, "
              f"IC95 [{bajo:+.4f}, {alto:+.4f}]) | AUC-ROC {fila['auc_roc']:.4f} | {segundos:.0f} s")
    print("\n" + hechos.sort_values("auc_pr", ascending=False).round(4).to_string(index=False))


if __name__ == "__main__":
    import argparse
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument("--folds", type=int, nargs="+")
    argumentos.add_argument("--candidatos", nargs="+")
    a = argumentos.parse_args()
    if a.folds:
        folds(a.candidatos or ["Extra Trees"], a.folds)
    else:
        main()
