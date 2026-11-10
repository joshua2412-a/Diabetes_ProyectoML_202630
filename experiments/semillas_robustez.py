"""
Sensibilidad, robustez y varianza por semilla de DES-CPL en el problema
principal (protocolo, sección 6; guía del Entregable 3, secciones 6.2 y 6.6).

Tres estudios, en este orden (del más barato al más caro):

1. **Sensibilidad** (fold 0, sin reentrenar). AUC-PR de DES-CPL en una
   rejilla de λ × k con la τ y la poda elegidas, y en un barrido de τ con el
   k y el λ elegidos. Usa las predicciones ya calculadas en
   ``results/des_cpl_60/``.
2. **Robustez** (5 folds). Se modifica solo el entrenamiento de cada fold
   externo y se evalúa en su fold de evaluación intacto:
   - ``ruido_05`` y ``ruido_10``: se invierte la etiqueta del 5 % y del 10 %
     de los encuentros de entrenamiento;
   - ``prevalencia_05``: se descartan positivos al azar hasta que sean el
     5 % del entrenamiento (en los datos originales son el 11,4 %);
   - ``tamano_25`` y ``tamano_50``: se entrena con el 25 % y el 50 % de los
     pacientes.

   Los modelos base conservan los hiperparámetros del capítulo 12 en cada
   fold (no se vuelven a buscar), y DES-CPL repite su búsqueda de 60
   evaluaciones. Así se aísla cómo responde cada método a la perturbación.
3. **Semillas** (5 folds). Diez semillas para la partición interna del
   conjunto de selección, los modelos base (incluido XGBoost, el mejor
   benchmark) y la búsqueda de Optuna: 42, 7, 2026, 1, 2, 3, 4, 5, 6 y 8. La
   semilla 42 reutiliza ``results/des_cpl_60/``.

Uso (raíz del repositorio):

    python experiments/semillas_robustez.py

Cada fold de cada escenario o semilla se guarda al terminarlo; si se
interrumpe, continúa donde quedó. Del orden de 7 a 9 horas en el equipo del
autor (unos 7 minutos por fold y escenario o semilla).

Produce
-------
results/sensibilidad_des_cpl.csv
results/robustez_des_cpl.csv
results/semillas_des_cpl.csv
(las predicciones intermedias quedan en results/robustez/ y results/semillas/,
ignoradas por git)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import configuracion as cfg                    # noqa: E402
from src import experimento_des_cpl as exd              # noqa: E402

PROCESADOS = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "results"
CARPETA_60 = RESULTADOS / "des_cpl_60"
RUTA_SENSIBILIDAD = RESULTADOS / "sensibilidad_des_cpl.csv"
RUTA_ROBUSTEZ = RESULTADOS / "robustez_des_cpl.csv"
RUTA_SEMILLAS = RESULTADOS / "semillas_des_cpl.csv"

ESCENARIOS = ("ruido_05", "ruido_10", "prevalencia_05", "tamano_25", "tamano_50")
SEMILLAS = (42, 7, 2026, 1, 2, 3, 4, 5, 6, 8)
FOLDS = range(5)


def cargar():
    roles = json.loads((PROCESADOS / "roles_variables.json").read_text(encoding="utf-8"))
    predictores = (roles["numericas"] + roles["ordinales"] + roles["binarias"]
                   + roles["categoricas"])
    train = pd.read_parquet(PROCESADOS / "diabetes_train.parquet")
    X, y, grupos = train[predictores], train[roles["objetivo"]], train[roles["identificador"]]
    mejores = exd.mejores_combinaciones(
        pd.read_csv(RESULTADOS / "tabla_maestra_clasificacion.csv"))
    mejores = exd.con_presupuesto(mejores, RESULTADOS / "convergencia_evaluaciones.csv",
                                  RESULTADOS / "convergencia_reajustes.csv", 60)
    return X, y, grupos, roles, mejores, exd.particiones_externas(X, y, grupos)


def _anexar(ruta, tabla):
    tabla.to_csv(ruta, mode="a", header=not ruta.exists(), index=False)


def _hechos(ruta, columnas):
    if not ruta.exists():
        return set()
    return set(map(tuple, pd.read_csv(ruta)[columnas].drop_duplicates().to_numpy().tolist()))


# ---------------------------------------------------------------------------
# 1. Sensibilidad
# ---------------------------------------------------------------------------

def sensibilidad(modelos):
    if RUTA_SENSIBILIDAD.exists():
        print("[omitido] sensibilidad: ya está en disco")
        return
    elegido = (pd.read_csv(RESULTADOS / "des_cpl_evaluacion_60.csv")
               .query("fold == 0 and método == 'DES-CPL'").iloc[0])
    k0, tau0, lam0 = int(elegido["k"]), float(elegido["tau"]), float(elegido["lambda"])
    poda0 = 4 if len(str(elegido["modelos_combinados"]).split(",")) == 4 else None
    seleccion = pd.read_csv(CARPETA_60 / "seleccion_fold0.csv")
    evaluado = pd.read_csv(CARPETA_60 / "evaluacion_fold0.csv")

    def evaluar(k, tau, lam):
        tabla = exd.evaluar_fold(seleccion, evaluado, modelos, ks=(k,), taus=(tau,),
                                 lambdas=(lam,), podas=(poda0,), n_evaluaciones=1,
                                 variantes=("DES-CPL",))
        fila = tabla.set_index("método").loc["DES-CPL"]
        return {"k": k, "tau": tau, "lambda": lam, "poda": poda0 or len(modelos),
                "auc_pr_seleccion": fila["auc_pr_seleccion"], "auc_pr": fila["auc_pr"],
                "auc_roc": fila["auc_roc"], "brier": fila["brier"],
                "elegido": (k, tau, lam) == (k0, tau0, lam0)}

    filas = []
    inicio = time.perf_counter()
    for lam in exd.LAMBDA_REJILLA:
        for k in exd.K_REJILLA:
            filas.append({"barrido": "lambda x k", **evaluar(k, tau0, lam)})
    for tau in exd.TAU_REJILLA:
        filas.append({"barrido": "tau", **evaluar(k0, float(tau), lam0)})
    pd.DataFrame(filas).to_csv(RUTA_SENSIBILIDAD, index=False)
    print(f"[listo] sensibilidad: {len(filas)} configuraciones "
          f"({(time.perf_counter() - inicio) / 60:.1f} min)")


# ---------------------------------------------------------------------------
# 2. Robustez
# ---------------------------------------------------------------------------

def perturbar(escenario, y, grupos, entrenamiento, fold):
    """Devuelve (y, índices de entrenamiento) del escenario; la evaluación no cambia."""
    rng = np.random.default_rng(cfg.SEMILLA + fold)
    if escenario.startswith("ruido"):
        proporcion = int(escenario.split("_")[1]) / 100
        invertidos = rng.choice(entrenamiento, size=round(proporcion * len(entrenamiento)),
                                replace=False)
        y = y.copy()
        y.iloc[invertidos] = 1 - y.iloc[invertidos]
        return y, entrenamiento
    if escenario.startswith("prevalencia"):
        objetivo = int(escenario.split("_")[1]) / 100
        positivos = entrenamiento[y.iloc[entrenamiento].to_numpy() == 1]
        negativos = entrenamiento[y.iloc[entrenamiento].to_numpy() == 0]
        n_pos = round(objetivo / (1 - objetivo) * len(negativos))
        conservados = rng.choice(positivos, size=n_pos, replace=False)
        return y, np.sort(np.concatenate([negativos, conservados]))
    if escenario.startswith("tamano"):
        proporcion = int(escenario.split("_")[1]) / 100
        pacientes = np.unique(grupos.iloc[entrenamiento])
        elegidos = rng.choice(pacientes, size=round(proporcion * len(pacientes)), replace=False)
        return y, entrenamiento[np.isin(grupos.iloc[entrenamiento], elegidos)]
    raise ValueError(escenario)


def robustez(X, y, grupos, roles, mejores, particiones):
    hechos = _hechos(RUTA_ROBUSTEZ, ["escenario", "fold"])
    for escenario in ESCENARIOS:
        for fold in FOLDS:
            if (escenario, fold) in hechos:
                continue
            inicio = time.perf_counter()
            entrenamiento, evaluacion = particiones[fold]
            y_e, tr = perturbar(escenario, y, grupos, entrenamiento, fold)
            razon = float((1 - y_e.iloc[tr].mean()) / y_e.iloc[tr].mean())
            seleccion, evaluado = exd.ejecutar_fold(
                fold, (tr, evaluacion), X, y_e, grupos, roles, mejores,
                RESULTADOS / "robustez" / escenario, razon_desbalance=razon, verbose=False)
            tabla = exd.evaluar_fold(seleccion, evaluado, list(mejores.index),
                                     semilla=cfg.SEMILLA + fold, variantes=("DES-CPL",))
            tabla.insert(0, "fold", fold)
            tabla.insert(0, "escenario", escenario)
            tabla["n_entrenamiento"] = len(tr)
            tabla["prevalencia_entrenamiento"] = float(y_e.iloc[tr].mean())
            _anexar(RUTA_ROBUSTEZ, tabla)
            r = tabla.set_index("método")["auc_pr"]
            print(f"[listo] robustez {escenario}, fold {fold}: DES-CPL {r['DES-CPL']:.4f}, "
                  f"XGBoost {r.get('xgboost', np.nan):.4f} ({(time.perf_counter() - inicio) / 60:.1f} min)")


# ---------------------------------------------------------------------------
# 3. Semillas
# ---------------------------------------------------------------------------

def semillas(X, y, grupos, roles, mejores, particiones):
    hechos = _hechos(RUTA_SEMILLAS, ["semilla", "fold"])
    razon = float((1 - y.mean()) / y.mean())
    for semilla in SEMILLAS:
        carpeta = CARPETA_60 if semilla == cfg.SEMILLA else RESULTADOS / "semillas" / f"s{semilla}"
        for fold in FOLDS:
            if (semilla, fold) in hechos:
                continue
            inicio = time.perf_counter()
            seleccion, evaluado = exd.ejecutar_fold(
                fold, particiones[fold], X, y, grupos, roles, mejores, carpeta,
                razon_desbalance=razon, semilla=semilla, verbose=False)
            tabla = exd.evaluar_fold(seleccion, evaluado, list(mejores.index),
                                     semilla=semilla + fold, variantes=("DES-CPL",))
            tabla.insert(0, "fold", fold)
            tabla.insert(0, "semilla", semilla)
            _anexar(RUTA_SEMILLAS, tabla)
            r = tabla.set_index("método")["auc_pr"]
            print(f"[listo] semilla {semilla}, fold {fold}: DES-CPL {r['DES-CPL']:.4f}, "
                  f"XGBoost {r.get('xgboost', np.nan):.4f} ({(time.perf_counter() - inicio) / 60:.1f} min)")


def main():
    cfg.N_JOBS = 4
    X, y, grupos, roles, mejores, particiones = cargar()
    sensibilidad(list(mejores.index))
    robustez(X, y, grupos, roles, mejores, particiones)
    semillas(X, y, grupos, roles, mejores, particiones)

    resumen = (pd.read_csv(RUTA_ROBUSTEZ).query("método in ['DES-CPL', 'xgboost']")
               .pivot_table(index="escenario", columns="método", values="auc_pr"))
    print("\nRobustez, AUC-PR media por escenario:\n", resumen.round(4))
    resumen = (pd.read_csv(RUTA_SEMILLAS).query("método in ['DES-CPL', 'xgboost']")
               .groupby(["método", "semilla"])["auc_pr"].mean().unstack(0))
    print("\nSemillas, AUC-PR media de los 5 folds:\n", resumen.round(4))
    print(resumen.agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
