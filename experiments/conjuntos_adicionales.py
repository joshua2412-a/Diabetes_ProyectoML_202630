"""
Evaluación en tres conjuntos de datos adicionales (protocolo, sección 3;
guía del Entregable 3, sección 6.3).

Conjuntos (UCI, elegidos antes de conocer el desempeño del modelo):

1. ``credito``: Default of Credit Card Clients (Yeh y Lien, 2009),
   30 000 registros, 22 % de positivos.
2. ``banco``: Bank Marketing (Moro et al., 2014), 45 211 registros, 11,7 %.
   Se elimina ``duration`` (duración de la llamada): solo se conoce después
   de la llamada y determina casi por completo la respuesta, por lo que los
   autores recomiendan excluirla en un modelo predictivo realista.
3. ``cdc``: CDC Diabetes Health Indicators (BRFSS 2015), 253 680 registros,
   13,9 %.

Para cada conjunto y cada uno de los 5 folds externos estratificados
(semilla 42), se repite el procedimiento del problema principal:

1. Random Search con 60 evaluaciones para cada uno de los siete modelos
   base (con su balanceo del problema principal) y para LightGBM y CatBoost,
   validación interna de 3 pliegues, métrica AUC-PR.
2. Predicciones fuera de pliegue de los modelos base (conjunto de selección
   de DES-CPL) y del fold de evaluación.
3. DES-CPL con 60 evaluaciones de Optuna, los ensambles estáticos y los dos
   benchmarks del estado del arte, evaluados en el fold externo.

Desviaciones declaradas respecto del protocolo (por cómputo; se reportan en
el capítulo de evaluación ampliada):

- La búsqueda de hiperparámetros de todos los modelos se hace sobre una
  submuestra estratificada de ``TAMANO_BUSQUEDA`` registros del
  entrenamiento del fold (el capítulo 12 ya buscaba en un cuarto de los
  pacientes en los modelos costosos). El reajuste final usa todo el
  entrenamiento del fold. El presupuesto, 60 evaluaciones, es igual para
  todos los modelos y para DES-CPL.
- CDC se usa con una submuestra estratificada de ``TAMANO_CDC`` registros:
  la memoria de las regiones de competencia crece con n·k·M, y con los
  253 680 registros triplicaría la del problema principal.
- Solo se evalúa la versión final de DES-CPL (sin ablaciones).

Uso (raíz del repositorio):

    python experiments/conjuntos_adicionales.py

La primera vez descarga los datos de UCI a ``data/raw/adicionales/``.
Guarda cada paso al terminarlo en ``results/adicionales/<conjunto>/`` y, si
se interrumpe, continúa donde quedó. Del orden de 8 horas en el equipo del
autor (4 núcleos), empezando por el conjunto más pequeño.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src import configuracion as cfg                                  # noqa: E402
from src import experimento_des_cpl as exd                            # noqa: E402
from src.balanceo import construir_pipeline                           # noqa: E402
from src.espacios import espacio_busqueda                             # noqa: E402
from src.optimizacion import optimizar                                # noqa: E402
from src.sota import ESPACIOS, MODELOS_SOTA, construir_pipeline_sota  # noqa: E402

DATOS = RAIZ / "data" / "raw" / "adicionales"
RESULTADOS = RAIZ / "results" / "adicionales"
PRESUPUESTO = 60
TAMANO_BUSQUEDA = 5_000
TAMANO_CDC = 100_000
FOLDS = 5

FUENTES = {
    "credito": "https://archive.ics.uci.edu/static/public/350/data.csv",
    "banco": "https://archive.ics.uci.edu/static/public/222/data.csv",
    "cdc": "https://archive.ics.uci.edu/static/public/891/data.csv",
}
ORDEN = ("credito", "banco", "cdc")          # del más pequeño al más grande


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

def descargar(nombre: str) -> Path:
    """Descarga el CSV de UCI si no está en disco y devuelve su ruta."""
    ruta = DATOS / f"{nombre}.csv"
    if not ruta.exists():
        DATOS.mkdir(parents=True, exist_ok=True)
        print(f"Descargando {nombre} de {FUENTES[nombre]}")
        with urllib.request.urlopen(FUENTES[nombre], timeout=300) as respuesta:
            ruta.write_bytes(respuesta.read())
    return ruta


def _roles(numericas=(), ordinales=(), binarias=(), categoricas=()):
    return {"numericas": list(numericas), "ordinales": list(ordinales),
            "binarias": list(binarias), "categoricas": list(categoricas),
            "objetivo": "y", "identificador": "grupo"}


def preparar(nombre: str):
    """Devuelve (X, y, roles) del conjunto, con nombres de columna en español."""
    datos = pd.read_csv(descargar(nombre))
    if nombre == "credito":
        pagos = {f"X{i}": f"estado_pago_{i - 5}" for i in range(6, 12)}
        facturas = {f"X{i}": f"factura_{i - 11}" for i in range(12, 18)}
        abonos = {f"X{i}": f"abono_{i - 17}" for i in range(18, 24)}
        datos = datos.rename(columns={"X1": "limite_credito", "X5": "edad", "X3": "educacion",
                                      "X4": "estado_civil", **pagos, **facturas, **abonos})
        datos["sexo_masculino"] = (datos["X2"] == 1).astype(int)
        datos["educacion"] = datos["educacion"].astype(str)
        datos["estado_civil"] = datos["estado_civil"].astype(str)
        y = datos["Y"].astype(int)
        roles = _roles(numericas=["limite_credito", "edad", *facturas.values(), *abonos.values()],
                       ordinales=list(pagos.values()), binarias=["sexo_masculino"],
                       categoricas=["educacion", "estado_civil"])
    elif nombre == "banco":
        datos = datos.drop(columns="duration")
        for columna in ("default", "housing", "loan"):
            datos[columna] = (datos[columna] == "yes").astype(int)
        categoricas = ["job", "marital", "education", "contact", "month", "poutcome"]
        datos[categoricas] = datos[categoricas].fillna("desconocido")
        y = (datos["y"] == "yes").astype(int)
        roles = _roles(numericas=["age", "balance", "day_of_week", "campaign", "pdays", "previous"],
                       binarias=["default", "housing", "loan"], categoricas=categoricas)
    elif nombre == "cdc":
        datos, _ = train_test_split(datos, train_size=TAMANO_CDC, stratify=datos["Diabetes_binary"],
                                    random_state=cfg.SEMILLA)
        datos = datos.reset_index(drop=True)
        y = datos["Diabetes_binary"].astype(int)
        binarias = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
                    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
                    "NoDocbcCost", "DiffWalk", "Sex"]
        roles = _roles(numericas=["BMI", "MentHlth", "PhysHlth"],
                       ordinales=["GenHlth", "Age", "Education", "Income"], binarias=binarias)
    else:
        raise ValueError(nombre)
    columnas = roles["numericas"] + roles["ordinales"] + roles["binarias"] + roles["categoricas"]
    return datos[columnas].reset_index(drop=True), y.reset_index(drop=True), roles


# ---------------------------------------------------------------------------
# Búsqueda de hiperparámetros
# ---------------------------------------------------------------------------

def buscar(pipeline, espacio, X, y, semilla):
    """Random Search con 60 evaluaciones sobre una submuestra estratificada."""
    if len(X) > TAMANO_BUSQUEDA:
        X, _, y, _ = train_test_split(X, y, train_size=TAMANO_BUSQUEDA, stratify=y,
                                      random_state=semilla)
    grupos = np.arange(len(X))
    interno = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=semilla)
    particiones = list(interno.split(X, y, groups=grupos))
    resultado = optimizar("random", pipeline, {}, espacio, X, y, particiones,
                          "average_precision", PRESUPUESTO, semilla=semilla,
                          paralelo_interno=True)
    return resultado["mejores_parametros"]


def configuraciones_fold(nombre, fold, X_tr, y_tr, roles, mejores, razon):
    """Mejores hiperparámetros de los nueve modelos en un fold (con reanudación)."""
    ruta = RESULTADOS / nombre / "configuraciones.json"
    todas = json.loads(ruta.read_text()) if ruta.exists() else {}
    tiempos = RESULTADOS / nombre / "tiempos_busqueda.csv"
    for modelo in [*mejores.index, *MODELOS_SOTA]:
        clave = f"{fold}|{modelo}"
        if clave in todas:
            continue
        inicio = time.perf_counter()
        if modelo in MODELOS_SOTA:
            pipeline, espacio = construir_pipeline_sota(modelo, roles), ESPACIOS[modelo]
        else:
            pipeline = construir_pipeline(modelo, mejores.loc[modelo, "balanceo"], roles,
                                          cfg.SEMILLA, razon)
            espacio = espacio_busqueda(modelo)
        todas[clave] = buscar(pipeline, espacio, X_tr, y_tr, cfg.SEMILLA + fold)
        segundos = time.perf_counter() - inicio
        ruta.write_text(json.dumps(todas, indent=2, default=str))
        pd.DataFrame([{"conjunto": nombre, "fold": fold, "modelo": modelo,
                       "evaluaciones": PRESUPUESTO, "tiempo_s": segundos}]).to_csv(
            tiempos, mode="a", header=not tiempos.exists(), index=False)
        print(f"[listo] {nombre}, fold {fold}: búsqueda de {modelo} ({segundos / 60:.1f} min)")
    return {m: todas[f"{fold}|{m}"] for m in [*mejores.index, *MODELOS_SOTA]}


# ---------------------------------------------------------------------------
# Un conjunto completo
# ---------------------------------------------------------------------------

def ejecutar_conjunto(nombre, mejores_principal):
    carpeta = RESULTADOS / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_eval = carpeta / "evaluacion.csv"
    hechos = pd.read_csv(ruta_eval) if ruta_eval.exists() else pd.DataFrame(columns=["fold"])

    X, y, roles = preparar(nombre)
    grupos = pd.Series(np.arange(len(X)))          # sin identificador: cada registro es su grupo
    print(f"\n=== {nombre}: {len(X)} registros, {X.shape[1]} variables, "
          f"{y.mean():.1%} de positivos ===")
    particiones = exd.particiones_externas(X, y, grupos, FOLDS, cfg.SEMILLA)

    for fold, (entrenamiento, evaluacion) in enumerate(particiones):
        if fold in set(hechos["fold"]):
            print(f"[omitido] {nombre}, fold {fold}: ya evaluado")
            continue
        X_tr, y_tr = X.iloc[entrenamiento], y.iloc[entrenamiento]
        razon = float((1 - y_tr.mean()) / y_tr.mean())
        config = configuraciones_fold(nombre, fold, X_tr, y_tr, roles, mejores_principal, razon)

        mejores = mejores_principal.copy()
        mejores["hiperparametros_por_fold"] = [
            {f: config[m] if f == fold else {} for f in range(FOLDS)} for m in mejores.index]
        mejores["auc_pr_por_fold"] = [[np.nan] * FOLDS for _ in mejores.index]
        seleccion, evaluado = exd.ejecutar_fold(
            fold, (entrenamiento, evaluacion), X, y, grupos, roles, mejores, carpeta,
            razon_desbalance=razon, verbose=False)

        inicio = time.perf_counter()
        tabla = exd.evaluar_fold(seleccion, evaluado, list(mejores.index),
                                 semilla=cfg.SEMILLA + fold, variantes=("DES-CPL",))
        t_des = time.perf_counter() - inicio

        filas = []
        for modelo in MODELOS_SOTA:
            inicio = time.perf_counter()
            estimador = clone(construir_pipeline_sota(modelo, roles)).set_params(**config[modelo])
            p = estimador.fit(X_tr, y_tr).predict_proba(X.iloc[evaluacion])[:, 1]
            filas.append({"método": modelo, "familia": "Estado del arte",
                          **exd._metricas(evaluado["y"], p), "tiempo_s": time.perf_counter() - inicio})
        tabla = pd.concat([tabla, pd.DataFrame(filas)], ignore_index=True)
        tabla.loc[tabla["método"] == "DES-CPL", "tiempo_s"] = t_des
        tabla.insert(0, "fold", fold)
        tabla.insert(0, "conjunto", nombre)
        hechos = pd.concat([hechos, tabla], ignore_index=True)
        hechos.to_csv(ruta_eval, index=False)

        resumen = tabla.set_index("método")["auc_pr"].sort_values(ascending=False)
        print(f"[listo] {nombre}, fold {fold}: DES-CPL {resumen['DES-CPL']:.4f}, "
              f"mejor {resumen.index[0]} {resumen.iloc[0]:.4f}")
    return hechos


def main():
    cfg.N_JOBS = 4
    maestra = pd.read_csv(RAIZ / "results" / "tabla_maestra_clasificacion.csv")
    mejores = exd.mejores_combinaciones(maestra)[["balanceo"]]
    for nombre in ORDEN:
        ejecutar_conjunto(nombre, mejores)
    todas = pd.concat([pd.read_csv(RESULTADOS / n / "evaluacion.csv") for n in ORDEN],
                      ignore_index=True)
    todas.to_csv(RESULTADOS / "evaluacion_conjuntos.csv", index=False)
    print("\nAUC-PR media por conjunto:")
    print(todas.pivot_table(index="método", columns="conjunto", values="auc_pr").round(4))


if __name__ == "__main__":
    main()
