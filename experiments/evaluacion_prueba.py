"""
Evaluación final en el conjunto de prueba (capítulo 13, sección 13.8).

Se ejecuta UNA sola vez, después de la evaluación definitiva en el bucle
externo (protocolo, sección 3). Reproduce en el conjunto completo el
procedimiento de cada fold:

1. **Configuración final de cada benchmark.** Random Search con 60
   evaluaciones sobre todo el entrenamiento (un cuarto de los pacientes en
   los modelos costosos, como en el capítulo 12), validación interna de 3
   pliegues agrupada por paciente.
2. **DES-CPL.** Predicciones fuera de pliegue de los siete modelos base sobre
   el entrenamiento (3 pliegues agrupados) como conjunto de selección,
   búsqueda de sus hiperparámetros con 60 evaluaciones de Optuna y
   predicción del conjunto de prueba.
3. **LightGBM y CatBoost.** Igual que el paso 1, reajustados con todo el
   entrenamiento.

Uso (raíz del repositorio):

    python experiments/evaluacion_prueba.py

Produce results/prueba_configuraciones.json, results/prueba_metricas.csv y
results/prueba_predicciones.csv (probabilidades de cada método en prueba,
más la predicción fuera de pliegue de DES-CPL en el entrenamiento, que se
usa como calibración de la predicción conforme). Continúa donde quedó si se
interrumpe. Del orden de una a dos horas.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import configuracion as cfg                                  # noqa: E402
from src import experimento_des_cpl as exd                            # noqa: E402
from src.balanceo import construir_pipeline                           # noqa: E402
from src.espacios import espacio_busqueda                             # noqa: E402
from src.optimizacion import optimizar                                # noqa: E402
from src.sota import ESPACIOS, MODELOS_SOTA, construir_pipeline_sota  # noqa: E402

RESULTADOS = RAIZ / "results"
PROCESADOS = RAIZ / "data" / "processed"
RUTA_CONFIG = RESULTADOS / "prueba_configuraciones.json"
RUTA_METRICAS = RESULTADOS / "prueba_metricas.csv"
RUTA_PREDICCIONES = RESULTADOS / "prueba_predicciones.csv"
RUTA_OOF = RESULTADOS / "prueba_des_cpl_fuera_de_pliegue.csv"
CARPETA = RESULTADOS / "des_cpl_prueba"
PRESUPUESTO = 60
FRACCION_COSTOSOS = 0.25


def submuestra(grupos, fraccion, semilla):
    aleatorio = np.random.default_rng(semilla)
    pacientes = grupos.drop_duplicates()
    elegidos = aleatorio.choice(pacientes.to_numpy(),
                                size=max(2, int(len(pacientes) * fraccion)), replace=False)
    return grupos.isin(elegidos).to_numpy()


def buscar(pipeline, espacio, X, y, grupos, fraccion):
    """Random Search con 60 evaluaciones, como el capítulo 12."""
    if fraccion < 1:
        m = submuestra(grupos, fraccion, cfg.SEMILLA)
        X, y, grupos = X[m], y[m], grupos[m]
    interno = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cfg.SEMILLA)
    particiones = list(interno.split(X, y, groups=grupos))
    resultado = optimizar("random", pipeline, {}, espacio, X, y, particiones,
                          "average_precision", PRESUPUESTO, semilla=cfg.SEMILLA,
                          paralelo_interno=True)
    return resultado["mejores_parametros"]


def main():
    cfg.N_JOBS = 4
    roles = json.loads((PROCESADOS / "roles_variables.json").read_text(encoding="utf-8"))
    predictores = (roles["numericas"] + roles["ordinales"] + roles["binarias"]
                   + roles["categoricas"])
    train = pd.read_parquet(PROCESADOS / "diabetes_train.parquet")
    prueba = pd.read_parquet(PROCESADOS / "diabetes_test.parquet")
    todo = pd.concat([train, prueba], ignore_index=True)
    X, y, grupos = (todo[predictores], todo[roles["objetivo"]],
                    todo[roles["identificador"]])
    idx_tr, idx_te = np.arange(len(train)), np.arange(len(train), len(todo))
    X_tr, y_tr, g_tr = X.iloc[idx_tr], y.iloc[idx_tr], grupos.iloc[idx_tr]
    razon = float((1 - y_tr.mean()) / y_tr.mean())

    maestra = pd.read_csv(RESULTADOS / "tabla_maestra_clasificacion.csv")
    mejores = exd.mejores_combinaciones(maestra)
    fraccion = (maestra.query("estado == 'completada'")
                .loc[lambda t: t.groupby("modelo")["auc_pr_media"].idxmax()]
                .set_index("modelo")["fraccion_busqueda"].astype(float))

    # 1. Configuración final de cada modelo base.
    configuraciones = json.loads(RUTA_CONFIG.read_text()) if RUTA_CONFIG.exists() else {}
    for modelo in mejores.index:
        if modelo in configuraciones:
            print(f"[omitido] configuración de {modelo}")
            continue
        inicio = time.perf_counter()
        pipeline = construir_pipeline(modelo, mejores.loc[modelo, "balanceo"], roles,
                                      cfg.SEMILLA, razon)
        configuraciones[modelo] = buscar(pipeline, espacio_busqueda(modelo), X_tr, y_tr,
                                         g_tr, fraccion[modelo])
        RUTA_CONFIG.write_text(json.dumps(configuraciones, indent=2, default=str))
        print(f"[listo] configuración de {modelo} ({(time.perf_counter() - inicio) / 60:.1f} min)")

    # 2. DES-CPL y modelos base, con el mismo código que cada fold externo:
    #    «evaluación» es ahora el conjunto de prueba.
    finales = mejores.copy()
    finales["hiperparametros_por_fold"] = [[configuraciones[m]] for m in finales.index]
    finales["auc_pr_por_fold"] = [[np.nan] for _ in finales.index]
    seleccion, evaluado = exd.ejecutar_fold(0, (idx_tr, idx_te), X, y, grupos, roles,
                                            finales, CARPETA, razon_desbalance=razon)
    tabla, pred = exd.evaluar_fold(seleccion, evaluado, list(mejores.index),
                                   semilla=cfg.SEMILLA, devolver_predicciones=True)
    pd.DataFrame({"y": seleccion["y"], "grupo": seleccion["grupo"],
                  "DES-CPL": pred.pop("DES-CPL (fuera de pliegue)")}).to_csv(RUTA_OOF, index=False)

    # 3. Benchmarks del estado del arte.
    for nombre in MODELOS_SOTA:
        if nombre not in configuraciones:
            inicio = time.perf_counter()
            configuraciones[nombre] = buscar(construir_pipeline_sota(nombre, roles),
                                             ESPACIOS[nombre], X_tr, y_tr, g_tr, FRACCION_COSTOSOS)
            RUTA_CONFIG.write_text(json.dumps(configuraciones, indent=2, default=str))
            print(f"[listo] configuración de {nombre} ({(time.perf_counter() - inicio) / 60:.1f} min)")
        modelo = clone(construir_pipeline_sota(nombre, roles)).set_params(**configuraciones[nombre])
        pred[nombre] = modelo.fit(X_tr, y_tr).predict_proba(X.iloc[idx_te])[:, 1]
        tabla = pd.concat([tabla, pd.DataFrame([{
            "método": nombre, "familia": "Estado del arte",
            **exd._metricas(evaluado["y"], pred[nombre])}])], ignore_index=True)

    tabla.to_csv(RUTA_METRICAS, index=False)
    pd.DataFrame({"y": evaluado["y"], "grupo": evaluado["grupo"], **pred}).to_csv(
        RUTA_PREDICCIONES, index=False, float_format="%.8g")
    print("\n" + tabla.sort_values("auc_pr", ascending=False)
          [["método", "familia", "auc_pr", "auc_roc", "brier"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
