"""
Experimento del modelo original DES-CPL (capítulo 13).

Calcula las predicciones de los siete modelos base en los folds externos
pedidos y evalúa DES-CPL contra ellos. Es lo mismo que ejecuta el
capítulo 13; este script existe para correrlo desde la terminal, sin
Jupyter, cuando el cálculo es largo.

Uso (desde la raíz del repositorio, entorno ml_proyecto):

    python experiments/des_cpl.py --folds 0 1 2 3 4                    # versión definitiva
    python experiments/des_cpl.py --folds 0 1 2 3 4 --presupuesto e2   # exploratoria

Presupuesto. Por defecto (``60``) cada modelo base usa, en cada fold, la
configuración que eligió Random Search con 60 evaluaciones en el estudio de
convergencia del capítulo 12, y DES-CPL busca sus hiperparámetros con 60
evaluaciones de Optuna: modelo original y benchmarks reciben el mismo
presupuesto. Con ``e2`` se usan los hiperparámetros del Entregable 2 (8-12
evaluaciones); es la versión exploratoria con la que se diseñó el modelo.

Lee primero lo que ya está en disco y calcula solo lo que falta: si se
interrumpe, la siguiente ejecución continúa con el modelo que faltaba. Para
rehacer un fold, borra sus archivos en results/des_cpl/.

Produce
-------
results/des_cpl_60/seleccion_fold<i>.csv   Predicciones fuera de pliegue
results/des_cpl_60/evaluacion_fold<i>.csv  Predicciones del fold de evaluación
results/des_cpl_60/tiempos.csv             Tiempo de cada modelo y fold
results/des_cpl_evaluacion_60.csv          Métricas de todos los métodos
(con --presupuesto e2: results/des_cpl/ y results/des_cpl_evaluacion.csv)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import configuracion as cfg                    # noqa: E402
from src import experimento_des_cpl as exd              # noqa: E402

PROCESADOS = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "results"
RUTAS = {"60": (RESULTADOS / "des_cpl_60", RESULTADOS / "des_cpl_evaluacion_60.csv"),
         "e2": (RESULTADOS / "des_cpl", RESULTADOS / "des_cpl_evaluacion.csv")}


def main(folds, presupuesto="60"):
    CARPETA, RUTA_EVALUACION = RUTAS[presupuesto]
    cfg.N_JOBS = 4
    roles = json.loads((PROCESADOS / "roles_variables.json").read_text(encoding="utf-8"))
    predictores = (roles["numericas"] + roles["ordinales"] + roles["binarias"]
                   + roles["categoricas"])
    train = pd.read_parquet(PROCESADOS / "diabetes_train.parquet")
    X = train[predictores]
    y = train[roles["objetivo"]]
    grupos = train[roles["identificador"]]
    razon = float((1 - y.mean()) / y.mean())

    mejores = exd.mejores_combinaciones(
        pd.read_csv(RESULTADOS / "tabla_maestra_clasificacion.csv"))
    if presupuesto == "60":
        mejores = exd.con_presupuesto(mejores, RESULTADOS / "convergencia_evaluaciones.csv",
                                      RESULTADOS / "convergencia_reajustes.csv", 60)
    particiones = exd.particiones_externas(X, y, grupos)

    for fold in folds:
        print(f"\n{'=' * 70}\nFold externo {fold}\n{'=' * 70}")
        inicio = time.perf_counter()
        exd.ejecutar_fold(fold, particiones[fold], X, y, grupos, roles,
                          mejores, CARPETA, razon_desbalance=razon)
        print(f"Fold {fold}: {(time.perf_counter() - inicio) / 60:.1f} min")

    tabla = exd.evaluar(CARPETA, folds, RUTA_EVALUACION)
    print(exd.progreso(CARPETA, folds).to_string(index=False))
    if not tabla.empty:
        columnas = ["fold", "método", "k", "tau", "lambda", "auc_pr_seleccion",
                    "auc_pr", "auc_roc", "brier"]
        print("\n" + tabla[tabla["fold"].isin(folds)][columnas]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    argumentos.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                            help="Folds externos a calcular (0 a 4)")
    argumentos.add_argument("--presupuesto", choices=["60", "e2"], default="60",
                            help="60 (definitivo) o e2 (exploratorio)")
    a = argumentos.parse_args()
    main(a.folds, a.presupuesto)
