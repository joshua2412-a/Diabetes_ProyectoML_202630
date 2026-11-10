"""
Artefactos del dashboard (Bloque B del Entregable 3).

Reentrena la mejor combinación de cada uno de los siete modelos base del
Entregable 2 sobre el conjunto de entrenamiento completo, la serializa y
genera todo lo que el dashboard lee: el dashboard nunca entrena ni busca
hiperparámetros, solo carga lo que deja este script en ``artifacts/``.

Uso (desde la raíz del repositorio, entorno ml_proyecto):

    python experiments/entrenar_artefactos.py

Como los capítulos del libro, lee primero lo que ya está en disco y calcula
solo lo que falta: si se interrumpe, la siguiente ejecución continúa. Para
rehacer un artefacto, borra su archivo.

Qué produce
-----------
artifacts/mejores_combinaciones.csv   Tabla maestra de la selección
artifacts/<modelo>.joblib             Pipeline completo (compress=3)
artifacts/<modelo>.metadata.json      Versiones, semilla, hiperparámetros,
                                      esquema de entrada, métricas y prueba
                                      de ida y vuelta
artifacts/predicciones_prueba.parquet Probabilidades de los 7 en prueba
artifacts/fondo_shap.parquet          Conjunto de fondo de la explicación
artifacts/shap_global.parquet         Valores de Shapley de una muestra
artifacts/eda_faltantes.csv           Faltantes de los datos crudos

Criterios (los mismos del Entregable 2, para que el dashboard no contradiga
al libro):
* Mejor combinación de cada modelo: mayor AUC-PR media del bucle externo,
  la regla del capítulo 11.
* Hiperparámetros finales: la configuración modal entre los cinco folds
  externos, la regla del capítulo 10.
* Umbral operativo: percentil 85 de las probabilidades en prueba (marcar
  al 15 % de mayor riesgo), la regla del capítulo 10. No usa etiquetas.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import configuracion as cfg                      # noqa: E402
from src.balanceo import construir_pipeline               # noqa: E402
from src.estadistica import diferencia_critica            # noqa: E402
from src.explicacion import shapley_muestra               # noqa: E402
from src.metricas import calcular_metricas                # noqa: E402

ARTEFACTOS = RAIZ / "artifacts"
PROCESADOS = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "results"

#: Tamaño del conjunto de fondo y de la muestra de la explicación global.
#: El fondo es pequeño porque cada predicción en vivo lo multiplica por
#: n_permutaciones * (variables + 1).
N_FONDO = 20
N_MUESTRA_SHAP = 150
N_PERMUTACIONES = 8
CAPACIDAD_SEGUIMIENTO = 0.15


def _versiones():
    """Versiones de las librerías que deben coincidir al cargar el pickle."""
    import imblearn
    import sklearn
    import xgboost
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "imbalanced-learn": imblearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "joblib": joblib.__version__}


def _configuracion_modal(fila):
    """Configuración más frecuente entre folds externos (capítulo 10)."""
    por_fold = json.loads(fila["hiperparametros_por_fold"])
    serializados = [json.dumps(p, sort_keys=True) for p in por_fold]
    mas_comun, frecuencia = Counter(serializados).most_common(1)[0]
    return json.loads(mas_comun), frecuencia, len(por_fold)


def _esquema(X):
    """Nombres, tipos y rangos válidos de cada variable de entrada.

    Es lo que usa el formulario de predicción para generar sus campos y
    validar lo que escribe el usuario.
    """
    esquema = {}
    for columna in X.columns:
        serie = X[columna]
        if pd.api.types.is_numeric_dtype(serie):
            esquema[columna] = {
                "tipo": "numerica",
                "entero": bool(pd.api.types.is_integer_dtype(serie)
                               or (serie % 1 == 0).all()),
                "min": float(serie.min()), "max": float(serie.max()),
                "mediana": float(serie.median()),
            }
        else:
            frecuencias = serie.astype(str).value_counts()
            esquema[columna] = {
                "tipo": "categorica",
                "niveles": frecuencias.index.tolist(),
                "moda": str(frecuencias.index[0]),
            }
    return esquema


def seleccionar(maestra):
    """Mejor combinación de cada modelo con su motivo y su chequeo de finalista."""
    import scikit_posthocs as sp

    completadas = maestra.query("estado == 'completada'").copy()
    mejores = completadas.loc[completadas.groupby("modelo")["auc_pr_media"].idxmax()]
    por_fold = pd.DataFrame({f["modelo"]: json.loads(f["auc_pr_por_fold"])
                             for _, f in mejores.iterrows()})
    rangos = por_fold.rank(axis=1, ascending=False).mean()
    nemenyi = sp.posthoc_nemenyi_friedman(por_fold)
    global_ = mejores.sort_values("auc_pr_media", ascending=False).iloc[0]["modelo"]
    cd = diferencia_critica(por_fold.shape[1], por_fold.shape[0])

    filas = []
    for _, f in mejores.sort_values("auc_pr_media", ascending=False).iterrows():
        propias = completadas[completadas["modelo"] == f["modelo"]].sort_values(
            "auc_pr_media", ascending=False)
        segunda = propias.iloc[1]
        p = 1.0 if f["modelo"] == global_ else float(nemenyi.loc[f["modelo"], global_])
        filas.append({
            "modelo": f["modelo"], "balanceo": f["balanceo"],
            "optimizador": f["optimizador"],
            "auc_pr_media": f["auc_pr_media"], "auc_pr_sd": f["auc_pr_sd"],
            "auc_roc_media": f["auc_roc_media"], "auc_roc_sd": f["auc_roc_sd"],
            "brier_media": f["brier_media"],
            "combinaciones_evaluadas": len(propias),
            "rango_medio": float(rangos[f["modelo"]]),
            "p_nemenyi_vs_mejor": p,
            "finalista": bool(rangos[f["modelo"]] - rangos[global_] <= cd),
            "motivo": (f"Mayor AUC-PR media del bucle externo entre sus "
                       f"{len(propias)} combinaciones; la segunda "
                       f"({segunda['balanceo']} + {segunda['optimizador']}) "
                       f"queda a {f['auc_pr_media'] - segunda['auc_pr_media']:.4f}"),
        })
    tabla = pd.DataFrame(filas)
    tabla.attrs["mejor_global"] = global_
    tabla.attrs["diferencia_critica"] = cd
    return tabla, mejores.set_index("modelo")


def entrenar(seleccion, mejores, X_train, y_train, X_prueba, y_prueba, razon):
    """Reentrena, serializa y verifica cada modelo; devuelve sus predicciones."""
    predicciones = {"y_real": y_prueba.to_numpy()}
    for modelo in seleccion["modelo"]:
        ruta = ARTEFACTOS / f"{modelo}.joblib"
        ruta_meta = ARTEFACTOS / f"{modelo}.metadata.json"
        if ruta.exists() and ruta_meta.exists():
            print(f"[omitido] {modelo}: ya existe")
            predicciones[modelo] = joblib.load(ruta).predict_proba(X_prueba)[:, 1]
            continue

        fila = mejores.loc[modelo]
        parametros, frecuencia, total = _configuracion_modal(fila)
        pipeline = construir_pipeline(modelo, fila["balanceo"], ROLES,
                                      cfg.SEMILLA, razon).set_params(**parametros)
        inicio = time.perf_counter()
        pipeline.fit(X_train, y_train)
        t_ajuste = time.perf_counter() - inicio

        inicio = time.perf_counter()
        prob = pipeline.predict_proba(X_prueba)[:, 1]
        t_inferencia = time.perf_counter() - inicio

        # compress=3: el plan gratuito de Render tiene poca memoria y GitHub
        # limita el tamaño por archivo.
        joblib.dump(pipeline, ruta, compress=3)
        recargado = joblib.load(ruta)
        # Random Forest promedia los árboles en paralelo y el orden de la suma
        # varía entre llamadas, así que la igualdad bit a bit no está garantizada;
        # se exige coincidencia hasta el redondeo de punto flotante.
        diferencia_max = float(np.max(np.abs(
            recargado.predict_proba(X_prueba)[:, 1] - prob)))
        ida_y_vuelta = diferencia_max <= 1e-9
        if not ida_y_vuelta:
            raise RuntimeError(f"{modelo}: el modelo serializado no reproduce las "
                               f"predicciones (diferencia máxima {diferencia_max:.2e})")

        umbral = float(np.quantile(prob, 1 - CAPACIDAD_SEGUIMIENTO))
        metadata = {
            "modelo": modelo, "balanceo": fila["balanceo"],
            "optimizador": fila["optimizador"], "semilla": cfg.SEMILLA,
            "versiones": _versiones(),
            "hiperparametros": parametros,
            "hiperparametros_criterio": (f"Configuración modal: elegida en "
                                         f"{frecuencia} de {total} folds externos"),
            "esquema": _esquema(X_train),
            "orden_columnas": list(X_train.columns),
            "metricas_bucle_externo": {
                m: {"media": float(fila[f"{m}_media"]), "sd": float(fila[f"{m}_sd"])}
                for m in ["auc_pr", "auc_roc", "brier", "f1", "recall", "precision"]},
            "metricas_prueba": calcular_metricas(y_prueba, prob),
            "metricas_prueba_umbral": calcular_metricas(y_prueba, prob, umbral),
            "umbral": umbral,
            "umbral_criterio": (f"Percentil {int((1 - CAPACIDAD_SEGUIMIENTO) * 100)} "
                                f"de las probabilidades en prueba: marca al "
                                f"{int(CAPACIDAD_SEGUIMIENTO * 100)} % de mayor riesgo"),
            "tiempo_ajuste_s": t_ajuste,
            "tiempo_inferencia_ms_por_caso": 1000 * t_inferencia / len(X_prueba),
            "tamano_mb": ruta.stat().st_size / 2**20,
            "ida_y_vuelta": ida_y_vuelta, "ida_y_vuelta_diferencia_max": diferencia_max,
        }
        ruta_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False,
                                        default=str), encoding="utf-8")
        predicciones[modelo] = prob
        print(f"[listo] {modelo}: AUC-PR prueba {metadata['metricas_prueba']['auc_pr']:.4f}, "
              f"{metadata['tamano_mb']:.1f} MB, ajuste {t_ajuste:.0f} s, ida y vuelta OK")
    return pd.DataFrame(predicciones)


def faltantes_crudos():
    """Faltantes de los datos crudos y la decisión que tomó el ETL."""
    # "None" en A1Cresult y max_glu_serum significa «prueba no realizada»,
    # no un faltante: solo "?" marca ausencia (capítulo 1).
    crudo = pd.read_csv(RAIZ / "data" / "raw" / "diabetic_data.csv",
                        na_values=["?"], keep_default_na=False, low_memory=False)
    decisiones = {
        "weight": "Eliminada: el faltante es masivo y no informativo (cap. 2.4)",
        "payer_code": "Eliminada: identifica al asegurador, no al paciente (cap. 2.4)",
        "medical_specialty": "Categoría explícita en «especialidad» (cap. 2.6)",
        "race": "Categoría explícita, sin imputar: atributo protegido (cap. 2.6.1)",
        "diag_1": "Categoría explícita al agrupar los códigos ICD-9 (cap. 2.5.1)",
        "diag_2": "Categoría explícita al agrupar los códigos ICD-9 (cap. 2.5.1)",
        "diag_3": "Categoría explícita al agrupar los códigos ICD-9 (cap. 2.5.1)",
    }
    faltan = crudo.isna().sum()
    faltan = faltan[faltan > 0].sort_values(ascending=False)
    return pd.DataFrame({
        "variable": faltan.index, "faltantes": faltan.values,
        "porcentaje": 100 * faltan.values / len(crudo),
        "decision": [decisiones.get(v, "") for v in faltan.index],
    })


def explicacion_global(seleccion, X_train, X_prueba):
    """Valores de Shapley de una muestra de prueba para cada modelo."""
    ruta_fondo = ARTEFACTOS / "fondo_shap.parquet"
    if ruta_fondo.exists():
        fondo = pd.read_parquet(ruta_fondo)
    else:
        fondo = X_train.sample(N_FONDO, random_state=cfg.SEMILLA)
        fondo.to_parquet(ruta_fondo)

    ruta = ARTEFACTOS / "shap_global.parquet"
    hechos = pd.read_parquet(ruta) if ruta.exists() else pd.DataFrame()
    muestra = X_prueba.sample(N_MUESTRA_SHAP, random_state=cfg.SEMILLA)
    for modelo in seleccion["modelo"]:
        if not hechos.empty and modelo in set(hechos["modelo"]):
            print(f"[omitido] SHAP global de {modelo}")
            continue
        inicio = time.perf_counter()
        pipeline = joblib.load(ARTEFACTOS / f"{modelo}.joblib")
        valores = shapley_muestra(lambda t: pipeline.predict_proba(t)[:, 1],
                                  muestra, fondo, N_PERMUTACIONES)
        largo = valores.reset_index(names="fila").melt(
            id_vars="fila", var_name="variable", value_name="shap")
        largo.insert(0, "modelo", modelo)
        hechos = pd.concat([hechos, largo], ignore_index=True)
        hechos.to_parquet(ruta)
        print(f"[listo] SHAP global de {modelo} en {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    cfg.N_JOBS = 4
    ARTEFACTOS.mkdir(exist_ok=True)
    ROLES = json.loads((PROCESADOS / "roles_variables.json").read_text(encoding="utf-8"))
    PREDICTORES = (ROLES["numericas"] + ROLES["ordinales"] + ROLES["binarias"]
                   + ROLES["categoricas"])
    train = pd.read_parquet(PROCESADOS / "diabetes_train.parquet")
    prueba = pd.read_parquet(PROCESADOS / "diabetes_test.parquet")
    X_train, y_train = train[PREDICTORES], train[ROLES["objetivo"]]
    X_prueba, y_prueba = prueba[PREDICTORES], prueba[ROLES["objetivo"]]
    razon = float((1 - y_train.mean()) / y_train.mean())

    maestra = pd.read_csv(RESULTADOS / "tabla_maestra_clasificacion.csv")
    seleccion, mejores = seleccionar(maestra)
    seleccion.to_csv(ARTEFACTOS / "mejores_combinaciones.csv", index=False)
    print(f"Mejor combinación global: {seleccion.attrs['mejor_global']}; "
          f"finalistas (no distinguibles por Nemenyi): "
          f"{', '.join(seleccion.query('finalista')['modelo'])}")

    predicciones = entrenar(seleccion, mejores, X_train, y_train, X_prueba,
                            y_prueba, razon)
    predicciones.to_parquet(ARTEFACTOS / "predicciones_prueba.parquet")

    if not (ARTEFACTOS / "eda_faltantes.csv").exists():
        faltantes_crudos().to_csv(ARTEFACTOS / "eda_faltantes.csv", index=False)

    explicacion_global(seleccion, X_train, X_prueba)
    print("Artefactos completos en", ARTEFACTOS)
