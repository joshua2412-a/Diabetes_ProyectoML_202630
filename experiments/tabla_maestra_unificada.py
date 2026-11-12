"""
Tabla maestra unificada de experimentos (guía del Entregable 3, entregable 6).

Reúne en un solo CSV, con una fila por método y conjunto de datos, todo lo
que se evaluó en el bucle externo de los Entregables 2 y 3:

- las 112 combinaciones del Entregable 2 (7 modelos × 4 balanceos × 4
  optimizadores, presupuesto 8-12);
- la mejor combinación de cada modelo con presupuesto de 60 (capítulo 12),
  LightGBM y CatBoost, los ensambles estáticos, DES-CPL y sus ablaciones
  (capítulo 13);
- los 13 métodos en los tres conjuntos adicionales (capítulo 16);
- DES-CPL y XGBoost con cada una de las diez semillas (capítulo 16).

Cada fila trae la media y la desviación del AUC-PR, la media del AUC-ROC y
del Brier, el AUC-PR de cada fold, el presupuesto de búsqueda, la semilla y,
cuando se registró, el tiempo.

Uso (raíz del repositorio): ``python experiments/tabla_maestra_unificada.py``
→ ``results/tabla_maestra_unificada.csv``. Solo lee resultados; tarda segundos.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "results"
SEMILLA = 42


def _resumir(tabla, columnas_grupo, **constantes):
    """Media, desviación y valores por fold de cada método del bucle externo."""
    filas = []
    for clave, grupo in tabla.sort_values("fold").groupby(columnas_grupo, sort=False):
        clave = clave if isinstance(clave, tuple) else (clave,)
        filas.append({**dict(zip(columnas_grupo, clave)), **constantes,
                      "auc_pr_media": grupo["auc_pr"].mean(), "auc_pr_sd": grupo["auc_pr"].std(),
                      "auc_roc_media": grupo["auc_roc"].mean(), "brier_media": grupo["brier"].mean(),
                      "folds": len(grupo),
                      "auc_pr_por_fold": json.dumps([round(v, 6) for v in grupo["auc_pr"]])})
    return pd.DataFrame(filas)


def main():
    partes = []

    e2 = pd.read_csv(RESULTADOS / "tabla_maestra_clasificacion.csv")
    partes.append(pd.DataFrame({
        "bloque": "E2: experimento combinatorio", "conjunto": "diabetes", "método": e2["modelo"],
        "balanceo": e2["balanceo"], "optimizador": e2["optimizador"],
        "evaluaciones": e2["presupuesto"], "semilla": e2["semilla"],
        "auc_pr_media": e2["auc_pr_media"], "auc_pr_sd": e2["auc_pr_sd"],
        "auc_roc_media": e2["auc_roc_media"], "brier_media": e2["brier_media"],
        "folds": e2["folds_externos"], "auc_pr_por_fold": e2["auc_pr_por_fold"],
        "tiempo_total_s": e2["tiempo_total_s"], "estado": e2["estado"]}))

    e3 = pd.read_csv(RESULTADOS / "des_cpl_evaluacion_60.csv")
    familias = {"Modelo base": "E3: modelos base con presupuesto 60 (cap. 12)",
                "Ensamble estático": "E3: ensambles estáticos", "Modelo original": "E3: modelo original",
                "Versión inicial": "E3: modelo original (versión inicial)", "Ablación": "E3: ablación"}
    for familia, bloque in familias.items():
        sub = e3[e3["familia"] == familia]
        partes.append(_resumir(sub, ["método"], bloque=bloque, conjunto="diabetes",
                               evaluaciones=60, semilla=SEMILLA))
    sota = pd.read_csv(RESULTADOS / "benchmarks_sota.csv").rename(columns={"modelo": "método"})
    resumen = _resumir(sota, ["método"], bloque="E3: estado del arte", conjunto="diabetes",
                       evaluaciones=60, semilla=SEMILLA)
    tiempos = sota.groupby("método")[["tiempo_busqueda_s", "tiempo_ajuste_s"]].sum().sum(axis=1)
    resumen["tiempo_total_s"] = resumen["método"].map(tiempos)
    partes.append(resumen)

    adicionales = pd.read_csv(RESULTADOS / "adicionales" / "evaluacion_conjuntos.csv")
    partes.append(_resumir(adicionales, ["conjunto", "método"], bloque="E3: conjuntos adicionales",
                           evaluaciones=60, semilla=SEMILLA))

    semillas = pd.read_csv(RESULTADOS / "semillas_des_cpl.csv")
    semillas = semillas[semillas["método"].isin(["DES-CPL", "xgboost"])]
    partes.append(_resumir(semillas, ["método", "semilla"], bloque="E3: varianza por semilla",
                           conjunto="diabetes", evaluaciones=60))

    robustez = pd.read_csv(RESULTADOS / "robustez_des_cpl.csv")
    robustez = robustez[robustez["familia"].isin(["Modelo base", "Ensamble estático", "Modelo original"])]
    resumen = _resumir(robustez, ["escenario", "método"], bloque="E3: robustez", conjunto="diabetes",
                       evaluaciones=60, semilla=SEMILLA)
    resumen["conjunto"] = "diabetes (" + resumen.pop("escenario") + ")"
    partes.append(resumen)

    tabla = pd.concat(partes, ignore_index=True)
    columnas = ["bloque", "conjunto", "método", "balanceo", "optimizador", "evaluaciones", "semilla",
                "auc_pr_media", "auc_pr_sd", "auc_roc_media", "brier_media", "folds",
                "auc_pr_por_fold", "tiempo_total_s", "estado"]
    tabla = tabla[columnas]
    tabla.to_csv(RESULTADOS / "tabla_maestra_unificada.csv", index=False)
    print(f"results/tabla_maestra_unificada.csv: {len(tabla)} filas")
    print(tabla.groupby("bloque").size().to_string())


if __name__ == "__main__":
    main()
