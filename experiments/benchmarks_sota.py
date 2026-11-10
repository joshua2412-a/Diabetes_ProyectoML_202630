"""
Benchmarks del estado del arte: LightGBM y CatBoost (capítulo 13, sección 13.5).

Ejecuta, para cada benchmark y cada fold externo del Entregable 2, el mismo
procedimiento que el estudio de convergencia del capítulo 12 aplicó a los
modelos base:

1. Submuestra de búsqueda con un cuarto de los pacientes del fold
   (generador con semilla 42 + fold, como el motor del Entregable 2).
2. Random Search con 60 configuraciones, validación interna de 3 pliegues
   agrupada por paciente, métrica AUC-PR.
3. Reajuste de la mejor configuración con todo el entrenamiento del fold y
   evaluación una sola vez en el fold externo.

Uso (raíz del repositorio; requiere `pip install lightgbm catboost`):

    python experiments/benchmarks_sota.py

Guarda cada (modelo, fold) al terminarlo en results/benchmarks_sota.csv; si
se interrumpe, continúa. LightGBM tarda minutos; CatBoost, del orden de
horas.
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

from src import configuracion as cfg                                   # noqa: E402
from src import experimento_des_cpl as exd                             # noqa: E402
from src.metricas import calcular_metricas                             # noqa: E402
from src.optimizacion import optimizar                                 # noqa: E402
from src.sota import (ESPACIOS, FRACCION_BUSQUEDA, MODELOS_SOTA,       # noqa: E402
                      construir_pipeline_sota)

RESULTADOS = RAIZ / "results"
RUTA = RESULTADOS / "benchmarks_sota.csv"
PRESUPUESTO = 60


def submuestra_busqueda(grupos_train, fraccion, semilla):
    """Máscara de la submuestra de búsqueda: una fracción de los pacientes."""
    aleatorio = np.random.default_rng(semilla)
    pacientes = grupos_train.drop_duplicates()
    elegidos = aleatorio.choice(pacientes.to_numpy(),
                                size=max(2, int(len(pacientes) * fraccion)), replace=False)
    return grupos_train.isin(elegidos).to_numpy()


def ejecutar(nombre, fold, particion, X, y, grupos, roles):
    entrenamiento, evaluacion = particion
    X_tr, y_tr, g_tr = X.iloc[entrenamiento], y.iloc[entrenamiento], grupos.iloc[entrenamiento]
    mascara = submuestra_busqueda(g_tr, FRACCION_BUSQUEDA, cfg.SEMILLA + fold)
    X_b, y_b, g_b = X_tr[mascara], y_tr[mascara], g_tr[mascara]
    interno = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cfg.SEMILLA)
    particiones_internas = list(interno.split(X_b, y_b, groups=g_b))

    pipeline = construir_pipeline_sota(nombre, roles)
    inicio = time.perf_counter()
    resultado = optimizar("random", pipeline, {}, ESPACIOS[nombre], X_b, y_b,
                          particiones_internas, "average_precision", PRESUPUESTO,
                          semilla=cfg.SEMILLA + fold, paralelo_interno=True)
    t_busqueda = time.perf_counter() - inicio

    final = clone(pipeline).set_params(**resultado["mejores_parametros"])
    inicio = time.perf_counter()
    final.fit(X_tr, y_tr)
    t_ajuste = time.perf_counter() - inicio
    inicio = time.perf_counter()
    p = final.predict_proba(X.iloc[evaluacion])[:, 1]
    t_inferencia = time.perf_counter() - inicio
    return {"modelo": nombre, "fold": fold, "presupuesto": PRESUPUESTO,
            "n_evaluaciones": len(resultado["evaluaciones"]),
            "puntaje_interno": float(max(e["puntaje"] for e in resultado["evaluaciones"])),
            "parametros": json.dumps(resultado["mejores_parametros"], default=str),
            **calcular_metricas(y.iloc[evaluacion], p),
            "tiempo_busqueda_s": t_busqueda, "tiempo_ajuste_s": t_ajuste,
            "tiempo_inferencia_s": t_inferencia}


def main():
    cfg.N_JOBS = 4
    roles = json.loads((RAIZ / "data" / "processed" / "roles_variables.json")
                       .read_text(encoding="utf-8"))
    predictores = (roles["numericas"] + roles["ordinales"] + roles["binarias"]
                   + roles["categoricas"])
    train = pd.read_parquet(RAIZ / "data" / "processed" / "diabetes_train.parquet")
    X, y, grupos = (train[predictores], train[roles["objetivo"]],
                    train[roles["identificador"]])
    particiones = exd.particiones_externas(X, y, grupos)
    hechos = pd.read_csv(RUTA) if RUTA.exists() else pd.DataFrame(columns=["modelo", "fold"])
    for nombre in MODELOS_SOTA:
        for fold, particion in enumerate(particiones):
            if ((hechos["modelo"] == nombre) & (hechos["fold"] == fold)).any():
                print(f"[omitido] {nombre}, fold {fold}")
                continue
            inicio = time.perf_counter()
            fila = ejecutar(nombre, fold, particion, X, y, grupos, roles)
            hechos = pd.concat([hechos, pd.DataFrame([fila])], ignore_index=True)
            hechos.to_csv(RUTA, index=False)
            print(f"[listo] {nombre}, fold {fold}: AUC-PR {fila['auc_pr']:.4f} "
                  f"({(time.perf_counter() - inicio) / 60:.1f} min)")
    print("\n" + hechos.groupby("modelo")["auc_pr"].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
