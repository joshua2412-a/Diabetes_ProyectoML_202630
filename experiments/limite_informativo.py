"""
¿Dónde está la información que falta? Estudio del límite explicativo (cap. 13).

DES-CPL y los ocho candidatos del cribado combinan o cambian los modelos,
pero todos ven las mismas 38 variables. Este estudio pregunta lo contrario:
si se añade información que el ETL descartó, ¿se mueve el techo?

Tres fuentes, cada una contra el mismo modelo de referencia:

1. **Códigos ICD-9 completos.** El ETL agrupa los diagnósticos en capítulos.
   Aquí se codifican los códigos completos (~700) con encogimiento
   jerárquico empírico-Bayes (capítulo -> categoría de 3 dígitos -> código;
   Micci-Barreca, 2001), calculado fuera de pliegue para no filtrar la
   etiqueta.
2. **Historial del paciente** dentro del conjunto: número de encuentros
   previos y si reingresó antes. Supone que ``encounter_id`` es cronológico.
3. **Etiqueta ordinal** (``NO`` < ``>30`` < ``<30``) en lugar de la binaria:
   un modelo de tres clases y un puntaje compartido de «cualquier
   reingreso» con corrección para ``<30``.

Es un estudio exploratorio con su propio protocolo, no el del Entregable 2:
parte del CSV crudo (mismas exclusiones del ETL: altas por fallecimiento u
hospicio), reserva el 20 % de los pacientes como en el capítulo 2 y compara
con validación cruzada de 5 pliegues agrupada por paciente sobre el 80 %
restante. El modelo es un HistGradientBoosting con categóricas nativas e
hiperparámetros fijos, igual para todas las variantes: lo que cambia entre
filas es solo la información.

Uso (raíz del repositorio):

    python experiments/limite_informativo.py

Guarda cada (variante, pliegue) al terminar en
results/limite_informativo.csv; si se interrumpe, continúa. Unos 15 min.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import configuracion as cfg      # noqa: E402

SEMILLA = cfg.SEMILLA
RUTA = RAIZ / "results" / "limite_informativo.csv"
CRUDO = RAIZ / "data" / "raw" / "diabetic_data.csv"

#: Altas por fallecimiento u hospicio: no pueden reingresar (capítulo 2).
EXCLUIDAS = [11, 13, 14, 19, 20, 21]
NUMERICAS = ["time_in_hospital", "num_lab_procedures", "num_procedures",
             "num_medications", "number_outpatient", "number_emergency",
             "number_inpatient", "number_diagnoses"]
CATEGORICAS = ["race", "gender", "age", "admission_type_id",
               "discharge_disposition_id", "admission_source_id",
               "medical_specialty", "A1Cresult", "max_glu_serum", "insulin",
               "change", "diabetesMed", "metformin"]
DIAGNOSTICOS = ["diag_1", "diag_2", "diag_3"]
#: Pseudo-observaciones del encogimiento en cada nivel de la jerarquía.
M_ENCOGIMIENTO = 20
EPS = 1e-6

CAPITULOS = [(1, 140, "infecciosas"), (140, 240, "neoplasias"),
             (240, 280, "endocrinas"), (280, 290, "sangre"),
             (290, 320, "mentales"), (320, 390, "nervioso"),
             (390, 460, "circulatorio"), (460, 520, "respiratorio"),
             (520, 580, "digestivo"), (580, 630, "genitourinario"),
             (630, 680, "embarazo"), (680, 710, "piel"),
             (710, 740, "musculoesqueletico"), (740, 760, "congenitas"),
             (760, 780, "perinatales"), (780, 800, "sintomas"),
             (800, 1000, "lesiones")]


def capitulo(codigo):
    """Capítulo ICD-9 de un código (V y E aparte; 250.x es diabetes)."""
    if pd.isna(codigo):
        return "NA"
    codigo = str(codigo)
    if codigo[0] in "VE":
        return codigo[0]
    valor = float(codigo)
    if 250 <= valor < 251:
        return "diabetes"
    for desde, hasta, nombre in CAPITULOS:
        if desde <= valor < hasta:
            return nombre
    return "otro"


def categoria(codigo):
    """Categoría de 3 dígitos (la parte entera del código)."""
    return "NA" if pd.isna(codigo) else str(codigo).split(".")[0]


def niveles(serie):
    """Los tres niveles de la jerarquía, de grueso a fino."""
    return [serie.map(capitulo).to_numpy(), serie.map(categoria).to_numpy(),
            serie.fillna("NA").astype(str).to_numpy()]


def codificar_jerarquico(y_ajuste, niveles_ajuste, niveles_aplicar,
                         m=M_ENCOGIMIENTO):
    """Codificación empírico-Bayes por la jerarquía ICD-9.

    La tasa de cada nodo se encoge hacia la de su padre con ``m``
    pseudo-observaciones: un código raro hereda la tasa de su categoría y
    esta la de su capítulo. Un nodo no visto al ajustar toma la tasa de su
    ancestro más cercano visto.
    """
    padre_ajuste = np.full(len(y_ajuste), y_ajuste.mean())
    salida = np.full(len(niveles_aplicar[0]), y_ajuste.mean())
    for nivel_aj, nivel_ap in zip(niveles_ajuste, niveles_aplicar):
        tabla = (pd.DataFrame({"k": nivel_aj, "y": y_ajuste, "padre": padre_ajuste})
                 .groupby("k").agg(s=("y", "sum"), n=("y", "size"), padre=("padre", "mean")))
        tasa = (tabla["s"] + m * tabla["padre"]) / (tabla["n"] + m)
        nuevo = pd.Series(nivel_ap).map(tasa).to_numpy(dtype=float)
        salida = np.where(np.isnan(nuevo), salida, nuevo)
        padre_ajuste = pd.Series(nivel_aj).map(tasa).to_numpy(dtype=float)
    return salida


def cargar():
    d = pd.read_csv(CRUDO, na_values="?", keep_default_na=False, low_memory=False)
    d = d[~d["discharge_disposition_id"].isin(EXCLUIDAS)].reset_index(drop=True)
    base = d[NUMERICAS].copy()
    for c in CATEGORICAS:
        base[c] = d[c].astype(str).astype("category")
    for c in DIAGNOSTICOS:
        base[f"{c}_capitulo"] = d[c].map(capitulo).astype("category")
    nivel = d["readmitted"].map({"NO": 0, ">30": 1, "<30": 2}).to_numpy()
    return d, base, nivel


def historial(d, y, indices):
    """Encuentros previos del paciente dentro de ``indices``, en orden de encounter_id."""
    t = pd.DataFrame({"g": d["patient_nbr"].to_numpy()[indices],
                      "o": d["encounter_id"].to_numpy()[indices],
                      "y": y[indices]}).reset_index(drop=True)
    orden = t.sort_values(["g", "o"])
    grupo = orden.groupby("g")
    orden["encuentros_previos"] = grupo.cumcount()
    orden["readmision_previa"] = grupo["y"].shift(1).fillna(-1)
    orden["readmisiones_previas"] = grupo["y"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    return orden.sort_index()[["encuentros_previos", "readmision_previa",
                               "readmisiones_previas"]]


def con_icd(d, y, grupos, ajuste, aplicar, X_aplicar, fuera_de_pliegue):
    """Añade la codificación jerárquica de los tres diagnósticos."""
    X = X_aplicar.copy()
    for c in DIAGNOSTICOS:
        if fuera_de_pliegue:
            valores = np.zeros(len(aplicar))
            interno = StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA)
            for a, b in interno.split(ajuste, y[ajuste], grupos[ajuste]):
                valores[b] = codificar_jerarquico(
                    y[ajuste[a]], niveles(d.loc[ajuste[a], c]), niveles(d.loc[ajuste[b], c]))
        else:
            valores = codificar_jerarquico(y[ajuste], niveles(d.loc[ajuste, c]),
                                           niveles(d.loc[aplicar, c]))
        X[f"{c}_icd"] = valores
    return X


def modelo():
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, categorical_features="from_dtype",
        early_stopping=True, random_state=SEMILLA)


def _p(X_a, y_a, X_b):
    return modelo().fit(X_a, y_a).predict_proba(X_b)[:, 1]


def variantes_fold(d, base, y, nivel, grupos, ia, ib):
    """AUC-PR de cada variante en un pliegue."""
    Xa = base.iloc[ia].reset_index(drop=True)
    Xb = base.iloc[ib].reset_index(drop=True)
    resultados = {"Referencia (variables del ETL)": _p(Xa, y[ia], Xb)}

    Xa_icd = con_icd(d, y, grupos, ia, ia, Xa, fuera_de_pliegue=True)
    Xb_icd = con_icd(d, y, grupos, ia, ib, Xb, fuera_de_pliegue=False)
    resultados["+ ICD-9 jerárquico"] = _p(Xa_icd, y[ia], Xb_icd)

    ha, hb = historial(d, y, ia), historial(d, y, ib)
    resultados["+ historial del paciente"] = _p(pd.concat([Xa, ha], axis=1), y[ia],
                                                pd.concat([Xb, hb], axis=1))
    resultados["+ ICD-9 jerárquico + historial"] = _p(
        pd.concat([Xa_icd, ha], axis=1), y[ia], pd.concat([Xb_icd, hb], axis=1))

    resultados["Etiqueta ordinal: tres clases"] = (
        modelo().fit(Xa, nivel[ia]).predict_proba(Xb)[:, 2])
    cualquiera = (nivel >= 1).astype(int)
    resultados["Etiqueta ordinal: solo «cualquier reingreso»"] = _p(Xa, cualquiera[ia], Xb)
    # Puntaje compartido + corrección para <30: apilamiento fuera de pliegue
    # de logit P(cualquier reingreso) y logit P(<30).
    Z = np.zeros((len(ia), 2))
    interno = StratifiedGroupKFold(3, shuffle=True, random_state=SEMILLA)
    for a, b in interno.split(ia, y[ia], grupos[ia]):
        Z[b, 0] = logit(np.clip(_p(Xa.iloc[a], cualquiera[ia][a], Xa.iloc[b]), EPS, 1 - EPS))
        Z[b, 1] = logit(np.clip(_p(Xa.iloc[a], y[ia][a], Xa.iloc[b]), EPS, 1 - EPS))
    meta = LogisticRegression().fit(Z, y[ia])
    Zb = np.column_stack([
        logit(np.clip(resultados["Etiqueta ordinal: solo «cualquier reingreso»"], EPS, 1 - EPS)),
        logit(np.clip(resultados["Referencia (variables del ETL)"], EPS, 1 - EPS))])
    resultados["Etiqueta ordinal: puntaje compartido + corrección"] = (
        meta.predict_proba(Zb)[:, 1])
    return {k: average_precision_score(y[ib], v) for k, v in resultados.items()}


def main():
    d, base, nivel = cargar()
    y = (nivel == 2).astype(int)
    grupos = d["patient_nbr"].to_numpy()
    # Reserva del 20 % de los pacientes, como el capítulo 2; no se usa aquí.
    entrenamiento, _ = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEMILLA)
                            .split(d, y, grupos))
    hechos = pd.read_csv(RUTA) if RUTA.exists() else pd.DataFrame(columns=["fold"])
    pliegues = StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(
        entrenamiento, y[entrenamiento], grupos[entrenamiento])
    for fold, (a, b) in enumerate(pliegues):
        if fold in set(hechos["fold"]):
            print(f"[omitido] pliegue {fold}")
            continue
        inicio = time.perf_counter()
        puntajes = variantes_fold(d, base, y, nivel, grupos,
                                  entrenamiento[a], entrenamiento[b])
        filas = pd.DataFrame([{"fold": fold, "variante": k, "auc_pr": v}
                              for k, v in puntajes.items()])
        hechos = pd.concat([hechos, filas], ignore_index=True)
        hechos.to_csv(RUTA, index=False)
        print(f"[listo] pliegue {fold} en {time.perf_counter() - inicio:.0f} s")
    tabla = hechos.pivot_table(index="variante", columns="fold", values="auc_pr")
    referencia = tabla.loc["Referencia (variables del ETL)"]
    tabla["media"] = tabla.mean(axis=1)
    tabla["dif. vs referencia"] = tabla["media"] - referencia.mean()
    tabla["pliegues a favor"] = (tabla[list(referencia.index)] > referencia).sum(axis=1)
    print("\n" + tabla.round(4).sort_values("media", ascending=False).to_string())


if __name__ == "__main__":
    main()
