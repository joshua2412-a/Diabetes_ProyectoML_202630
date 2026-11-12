"""
Artefacto del modelo original, su explicación y la tabla de presupuesto
(guía del Entregable 3, secciones 4.2, 6.2 y 6.6; contrato del dashboard en
CLAUDE.md, sección 7.2).

Hace, en orden y guardando cada paso al terminarlo:

1. **Entrena DES-CPL con todo el entrenamiento**, exactamente con la
   configuración de la evaluación en prueba (sección 13.8):
   - los cuatro modelos que eligió la poda, con los hiperparámetros de su
     Random Search de 60 evaluaciones (``results/prueba_configuraciones.json``);
   - k, τ y λ elegidos por la búsqueda de 60 evaluaciones de Optuna
     (``results/prueba_metricas.csv``).

   Se le pasan solo los cuatro modelos podados: la poda ya se decidió con
   estos mismos datos, y guardar los otros tres (con sus modelos de pliegue)
   solo haría el artefacto más pesado.
2. **Lo serializa** en ``artifacts/des_cpl.joblib`` y comprueba la ida y
   vuelta: el modelo recargado debe reproducir sus predicciones. Además
   informa la diferencia con las predicciones de la sección 13.8.
3. **Integra el modelo al dashboard**: metadatos con las mismas claves que
   los benchmarks, su fila al final de ``mejores_combinaciones.csv``, su
   columna en ``predicciones_prueba.parquet`` y los archivos de
   ``artifacts/modelo_original/`` que llenan la pestaña «Modelo original».
4. **Explicación global**: valores de Shapley de los mismos 150 casos de
   prueba y el mismo fondo que los benchmarks, añadidos a
   ``shap_global.parquet``. Así se compara directamente con XGBoost.
5. **Mide el costo propio de DES-CPL** (su búsqueda de 60 evaluaciones sobre
   el conjunto de selección) y arma la tabla de presupuesto y tiempos de
   todos los modelos en ``results/presupuesto_tiempos.csv``.

Uso (raíz del repositorio, entorno ml_proyecto):

    python experiments/artefacto_des_cpl.py

Del orden de 15 a 25 minutos. Si se interrumpe, continúa donde quedó.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import configuracion as cfg                      # noqa: E402
from src import experimento_des_cpl as exd                # noqa: E402
from src.balanceo import construir_pipeline               # noqa: E402
from src.des_cpl import DESCPL                            # noqa: E402
from src.explicacion import shapley_muestra               # noqa: E402
from src.metricas import calcular_metricas                # noqa: E402

ARTEFACTOS = RAIZ / "artifacts"
ORIGINAL = ARTEFACTOS / "modelo_original"
PROCESADOS = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "results"
CLAVE = "des_cpl"
RUTA = ARTEFACTOS / f"{CLAVE}.joblib"
RUTA_META = ARTEFACTOS / f"{CLAVE}.metadata.json"
RUTA_COSTO = RESULTADOS / "des_cpl_costo_propio.json"

# Los mismos valores que experiments/entrenar_artefactos.py.
N_MUESTRA_SHAP = 150
N_PERMUTACIONES = 8
CAPACIDAD_SEGUIMIENTO = 0.15


def cargar_datos():
    roles = json.loads((PROCESADOS / "roles_variables.json").read_text(encoding="utf-8"))
    predictores = roles["numericas"] + roles["ordinales"] + roles["binarias"] + roles["categoricas"]
    train = pd.read_parquet(PROCESADOS / "diabetes_train.parquet")
    prueba = pd.read_parquet(PROCESADOS / "diabetes_test.parquet")
    return (roles, train[predictores], train[roles["objetivo"]], train[roles["identificador"]],
            prueba[predictores], prueba[roles["objetivo"]])


def configuracion_13_8():
    """Configuración de DES-CPL y de sus modelos base en la sección 13.8."""
    fila = pd.read_csv(RESULTADOS / "prueba_metricas.csv").query("método == 'DES-CPL'").iloc[0]
    modelos = [m.strip() for m in str(fila["modelos_combinados"]).split(",")]
    hiper = {"k": int(fila["k"]), "tau": float(fila["tau"]), "encogimiento": float(fila["lambda"])}
    base = json.loads((RESULTADOS / "prueba_configuraciones.json").read_text())
    return modelos, hiper, {m: base[m] for m in modelos}


def _esquema(X):
    """Igual que en experiments/entrenar_artefactos.py: lo usa el formulario del dashboard."""
    esquema = {}
    for columna in X.columns:
        serie = X[columna]
        if pd.api.types.is_numeric_dtype(serie):
            esquema[columna] = {"tipo": "numerica",
                                "entero": bool(pd.api.types.is_integer_dtype(serie)
                                               or (serie % 1 == 0).all()),
                                "min": float(serie.min()), "max": float(serie.max()),
                                "mediana": float(serie.median())}
        else:
            frecuencias = serie.astype(str).value_counts()
            esquema[columna] = {"tipo": "categorica", "niveles": frecuencias.index.tolist(),
                                "moda": str(frecuencias.index[0])}
    return esquema


def _versiones():
    """Versiones de las librerías que deben coincidir al cargar el pickle."""
    from importlib.metadata import PackageNotFoundError, version
    versiones = {"python": platform.python_version()}
    for paquete in ["scikit-learn", "xgboost", "imbalanced-learn", "scipy", "numpy", "pandas",
                    "joblib"]:
        try:
            versiones[paquete] = version(paquete)
        except PackageNotFoundError:
            versiones[paquete] = None
    return versiones


# ---------------------------------------------------------------------------
# 1-3. Entrenamiento, serialización e integración al dashboard
# ---------------------------------------------------------------------------

def entrenar(roles, X, y, g, X_prueba, y_prueba):
    """Ajusta, serializa y verifica DES-CPL; devuelve sus probabilidades en prueba."""
    if RUTA.exists() and RUTA_META.exists():
        print(f"[omitido] {CLAVE}: ya existe")
        return joblib.load(RUTA).predict_proba(X_prueba)[:, 1]

    modelos, hiper, base = configuracion_13_8()
    balanceos = exd.mejores_combinaciones(
        pd.read_csv(RESULTADOS / "tabla_maestra_clasificacion.csv"))["balanceo"]
    razon = float((1 - y.mean()) / y.mean())
    estimadores = [(m, construir_pipeline(m, balanceos[m], roles, cfg.SEMILLA, razon)
                    .set_params(**base[m])) for m in modelos]
    modelo = DESCPL(estimadores, n_pliegues=3, n_modelos=None, semilla=cfg.SEMILLA, **hiper)
    print(f"Entrenando DES-CPL con {', '.join(modelos)}; {hiper}")

    inicio = time.perf_counter()
    modelo.fit(X, y, grupos=g)
    t_ajuste = time.perf_counter() - inicio
    inicio = time.perf_counter()
    prob = modelo.predict_proba(X_prueba)[:, 1]
    t_inferencia = time.perf_counter() - inicio

    # compress=3, como los benchmarks: Render gratuito tiene 512 MB.
    joblib.dump(modelo, RUTA, compress=3)
    diferencia_max = float(np.max(np.abs(joblib.load(RUTA).predict_proba(X_prueba)[:, 1] - prob)))
    if diferencia_max > 1e-9:
        raise RuntimeError(f"El modelo serializado no reproduce sus predicciones "
                           f"(diferencia máxima {diferencia_max:.2e})")
    seccion = pd.read_csv(RESULTADOS / "prueba_predicciones.csv")["DES-CPL"].to_numpy()
    diferencia_13_8 = float(np.max(np.abs(seccion - prob)))

    externo = pd.read_csv(RESULTADOS / "des_cpl_evaluacion_60.csv").query("método == 'DES-CPL'")
    umbral = float(np.quantile(prob, 1 - CAPACIDAD_SEGUIMIENTO))
    metadata = {
        "modelo": CLAVE, "balanceo": "por modelo base", "optimizador": "optuna",
        "semilla": cfg.SEMILLA, "versiones": _versiones(),
        "modelos_base": {m: {"balanceo": balanceos[m], "hiperparametros": base[m]} for m in modelos},
        "hiperparametros": {**hiper, "n_pliegues": 3, "competencia": "brier",
                            "ponderacion": "prevalencia", "poda": len(modelos)},
        "hiperparametros_criterio": ("Búsqueda bayesiana (Optuna, TPE) de 60 evaluaciones sobre el "
                                     "conjunto de selección del entrenamiento completo (sección 13.8)"),
        "esquema": _esquema(X), "orden_columnas": list(X.columns),
        "metricas_bucle_externo": {m: {"media": float(externo[m].mean()), "sd": float(externo[m].std())}
                                   for m in ["auc_pr", "auc_roc", "brier"]},
        "metricas_prueba": calcular_metricas(y_prueba, prob),
        "metricas_prueba_umbral": calcular_metricas(y_prueba, prob, umbral),
        "umbral": umbral,
        "umbral_criterio": (f"Percentil {int((1 - CAPACIDAD_SEGUIMIENTO) * 100)} de las "
                            f"probabilidades en prueba: marca al {int(CAPACIDAD_SEGUIMIENTO * 100)} % "
                            f"de mayor riesgo"),
        "tiempo_ajuste_s": t_ajuste,
        "tiempo_inferencia_ms_por_caso": 1000 * t_inferencia / len(X_prueba),
        "tamano_mb": RUTA.stat().st_size / 2**20,
        "ida_y_vuelta": True, "ida_y_vuelta_diferencia_max": diferencia_max,
        "diferencia_con_seccion_13_8": diferencia_13_8,
    }
    RUTA_META.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")
    print(f"[listo] {CLAVE}: AUC-PR prueba {metadata['metricas_prueba']['auc_pr']:.4f}, "
          f"{metadata['tamano_mb']:.1f} MB, ajuste {t_ajuste / 60:.1f} min, ida y vuelta OK; "
          f"diferencia con la sección 13.8: {diferencia_13_8:.1e}")
    return prob


def integrar_dashboard(prob):
    """Fila en la tabla de líderes, columna de predicciones y pestaña «Modelo original»."""
    ruta = ARTEFACTOS / "mejores_combinaciones.csv"
    mejores = pd.read_csv(ruta)
    if CLAVE not in set(mejores["modelo"]):
        externo = pd.read_csv(RESULTADOS / "des_cpl_evaluacion_60.csv").query("método == 'DES-CPL'")
        fila = {"modelo": CLAVE, "balanceo": "por modelo base", "optimizador": "optuna",
                "auc_pr_media": externo["auc_pr"].mean(), "auc_pr_sd": externo["auc_pr"].std(),
                "auc_roc_media": externo["auc_roc"].mean(), "auc_roc_sd": externo["auc_roc"].std(),
                "brier_media": externo["brier"].mean(), "combinaciones_evaluadas": 60,
                "rango_medio": np.nan, "p_nemenyi_vs_mejor": np.nan, "finalista": True,
                "motivo": ("Modelo original del Entregable 3. Combina árbol, Random Forest, XGBoost y "
                           "SVM; Nemenyi no lo separa del mejor benchmark (capítulo 13)")}
        # Va al final: la tabla de líderes toma la primera fila como la mejor del E2.
        pd.concat([mejores, pd.DataFrame([fila])], ignore_index=True).to_csv(ruta, index=False)

    ruta = ARTEFACTOS / "predicciones_prueba.parquet"
    predicciones = pd.read_parquet(ruta)
    predicciones[CLAVE] = prob
    predicciones.to_parquet(ruta)

    ORIGINAL.mkdir(exist_ok=True)
    externo = pd.read_csv(RESULTADOS / "des_cpl_evaluacion_60.csv").query("método == 'DES-CPL'")
    (ORIGINAL / "por_fold.json").write_text(json.dumps(
        {"nombre": CLAVE, "auc_pr_por_fold": externo.sort_values("fold")["auc_pr"].tolist()}),
        encoding="utf-8")

    pruebas = pd.read_csv(RESULTADOS / "pruebas_hipotesis_modelo_original.csv", index_col=0)
    for c in ["p crudo", "p ajustado (Holm)"]:
        pruebas[c] = pruebas[c].map(lambda v: f"{v:.2e}" if v < 1e-3 else f"{v:.3f}")
    (ORIGINAL / "pruebas_hipotesis.json").write_text(
        json.dumps(pruebas.to_dict("records"), ensure_ascii=False, indent=2), encoding="utf-8")

    (ORIGINAL / "veredicto.json").write_text(json.dumps({
        "titulo": "Veredicto: R1 y R2 se cumplen, R3 no",
        "texto": ("DES-CPL es original (no se encontró selección dinámica de ensambles en la "
                  "readmisión de pacientes con diabetes; capítulo 14) y está implementado por los "
                  "autores (src/des_cpl.py). Empata con el mejor benchmark, pero no lo supera de "
                  "forma significativa en ninguna de las tres pruebas (capítulos 13 y 16): el tope "
                  "de la nota es 4.0. Es más estable que XGBoost frente a la semilla y a "
                  "perturbaciones de los datos."),
        "superior": False}, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = json.loads(RUTA_META.read_text(encoding="utf-8"))
    (ORIGINAL / "resumen.md").write_text(_resumen(metadata, externo), encoding="utf-8")
    print("[listo] integración con el dashboard (tabla de líderes, predicciones, modelo_original/)")


def _resumen(metadata, externo):
    h = metadata["hiperparametros"]
    prueba = metadata["metricas_prueba"]
    return f"""### DES-CPL: selección dinámica de ensambles por competencia probabilística local

**Idea.** Los mejores modelos del benchmark empatan en promedio. DES-CPL decide, para cada paciente, cuánto confiar en cada modelo según la calidad de sus probabilidades (puntuación de Brier) en pacientes parecidos, y encoge esa competencia local hacia la global cuando la local es ruidosa.

**Algoritmo** (capítulo 15):
1. Predicciones fuera de pliegue de los modelos base, en 3 pliegues agrupados por paciente.
2. Poda a los modelos de mayor AUC-PR fuera de pliegue: {", ".join(metadata["modelos_base"])}.
3. Región de competencia: los $k$ vecinos positivos y los $k$ negativos en el espacio de logits de los modelos, sin el propio paciente.
4. Competencia: el Brier local, reponderado por la prevalencia local estimada.
5. Encogimiento hacia la competencia global: $\\hat C = (1-\\lambda)\\hat C^{{loc}} + \\lambda C^{{glob}}$.
6. Pesos $w_m \\propto \\exp(-\\hat C_m/\\tau)$ y predicción $\\hat p = \\sum_m w_m p_m$.

**Configuración del artefacto:** $k = {h["k"]}$, $\\tau = {h["tau"]:g}$, $\\lambda = {h["encogimiento"]}$, elegidos con 60 evaluaciones de Optuna.

**Desempeño:**
- Bucle externo, 5 folds: AUC-PR {externo["auc_pr"].mean():.4f} ± {externo["auc_pr"].std():.4f}.
- Prueba: AUC-PR {prueba["auc_pr"]:.4f}, AUC-ROC {prueba["auc_roc"]:.4f}.
- Supera de forma significativa a 6 de los 7 modelos base en prueba (DeLong con Holm) y empata con XGBoost, LightGBM y CatBoost.

**Evaluación ampliada** (capítulo 16):
- Es 3.º en rango medio entre 13 métodos y 4 conjuntos de datos, por delante de todos los benchmarks individuales.
- Pierde 1.6 a 1.9 veces menos que XGBoost ante ruido en las etiquetas, menos prevalencia y menos datos.
- En 10 semillas su AUC-PR es 0.2270 ± 0.0007, frente a 0.2251 ± 0.0017 de XGBoost.
"""


# ---------------------------------------------------------------------------
# 4. Explicación global
# ---------------------------------------------------------------------------

def explicacion_global(X_prueba):
    ruta = ARTEFACTOS / "shap_global.parquet"
    hechos = pd.read_parquet(ruta)
    if CLAVE in set(hechos["modelo"]):
        print(f"[omitido] SHAP global de {CLAVE}")
        return
    fondo = pd.read_parquet(ARTEFACTOS / "fondo_shap.parquet")
    muestra = X_prueba.sample(N_MUESTRA_SHAP, random_state=cfg.SEMILLA)
    modelo = joblib.load(RUTA)
    inicio = time.perf_counter()
    valores = shapley_muestra(lambda t: modelo.predict_proba(t)[:, 1], muestra,
                              fondo[X_prueba.columns], N_PERMUTACIONES)
    largo = valores.reset_index(names="fila").melt(id_vars="fila", var_name="variable",
                                                    value_name="shap")
    largo.insert(0, "modelo", CLAVE)
    pd.concat([hechos, largo], ignore_index=True).to_parquet(ruta)
    print(f"[listo] SHAP global de {CLAVE} en {time.perf_counter() - inicio:.0f} s")


# ---------------------------------------------------------------------------
# 5. Costo propio y tabla de presupuesto
# ---------------------------------------------------------------------------

def costo_propio():
    """Tiempo de la búsqueda de 60 evaluaciones de DES-CPL sobre su conjunto de selección.

    Es lo que DES-CPL añade sobre los modelos base que combina: no reentrena
    nada, solo consulta vecinos y combina probabilidades.
    """
    if RUTA_COSTO.exists():
        return json.loads(RUTA_COSTO.read_text())
    modelo = joblib.load(RUTA)
    inicio = time.perf_counter()
    exd._buscar(lambda k, lam, tau: modelo.puntuar_seleccion(k=k, tau=tau, encogimiento=lam),
                {"k": list(exd.K_REJILLA), "lam": list(exd.LAMBDA_REJILLA),
                 "tau": [float(t) for t in exd.TAU_REJILLA]}, exd.PRESUPUESTO, cfg.SEMILLA)
    costo = {"evaluaciones": exd.PRESUPUESTO, "tiempo_busqueda_s": time.perf_counter() - inicio,
             "n_seleccion": int(len(modelo.y_seleccion_)), "modelos": modelo.nombres_}
    RUTA_COSTO.write_text(json.dumps(costo, indent=2))
    print(f"[listo] búsqueda propia de DES-CPL: {costo['tiempo_busqueda_s'] / 60:.1f} min")
    return costo


def tabla_presupuesto(costo):
    """Evaluaciones, tiempos y tamaño de cada modelo (guía, sección 6.2)."""
    filas = []
    busqueda = (pd.read_csv(RESULTADOS / "convergencia_evaluaciones.csv")
                .groupby(["modelo", "fold"])["tiempo_s"].sum().groupby("modelo").mean())
    for modelo, segundos in busqueda.items():
        meta = json.loads((ARTEFACTOS / f"{modelo}.metadata.json").read_text(encoding="utf-8"))
        filas.append({"modelo": modelo, "familia": "Modelo base", "evaluaciones": 60,
                      "busqueda_min_por_fold": segundos / 60,
                      "ajuste_final_s": meta["tiempo_ajuste_s"],
                      "inferencia_ms_por_caso": meta["tiempo_inferencia_ms_por_caso"],
                      "artefacto_mb": meta["tamano_mb"]})
    sota = pd.read_csv(RESULTADOS / "benchmarks_sota.csv")
    # El tiempo de inferencia de los SOTA es el de un fold externo (1/5 del entrenamiento).
    n_fold = len(pd.read_parquet(PROCESADOS / "diabetes_train.parquet")) / 5
    for modelo, grupo in sota.groupby("modelo"):
        filas.append({"modelo": modelo, "familia": "Estado del arte", "evaluaciones": 60,
                      "busqueda_min_por_fold": grupo["tiempo_busqueda_s"].mean() / 60,
                      "ajuste_final_s": grupo["tiempo_ajuste_s"].mean(),
                      "inferencia_ms_por_caso": 1000 * grupo["tiempo_inferencia_s"].mean() / n_fold,
                      "artefacto_mb": np.nan})
    meta = json.loads(RUTA_META.read_text(encoding="utf-8"))
    tiempos = pd.read_csv(RESULTADOS / "des_cpl_60" / "tiempos.csv")
    filas.append({"modelo": CLAVE, "familia": "Modelo original", "evaluaciones": costo["evaluaciones"],
                  "busqueda_min_por_fold": costo["tiempo_busqueda_s"] / 60,
                  "ajuste_final_s": meta["tiempo_ajuste_s"],
                  "inferencia_ms_por_caso": meta["tiempo_inferencia_ms_por_caso"],
                  "artefacto_mb": meta["tamano_mb"],
                  "conjunto_seleccion_min_por_fold": (tiempos[["t_pliegues_s", "t_final_s"]].sum(axis=1)
                                                      .groupby(tiempos["fold"]).sum().mean() / 60)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RESULTADOS / "presupuesto_tiempos.csv", index=False)
    print("\nPresupuesto y tiempos (hardware: 4 núcleos físicos, N_JOBS = 4):")
    print(tabla.round(3).to_string(index=False))


def main():
    cfg.N_JOBS = 4
    roles, X, y, g, X_prueba, y_prueba = cargar_datos()
    prob = entrenar(roles, X, y, g, X_prueba, y_prueba)
    integrar_dashboard(prob)
    explicacion_global(X_prueba)
    tabla_presupuesto(costo_propio())


if __name__ == "__main__":
    main()
