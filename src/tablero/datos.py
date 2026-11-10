"""
Carga de datos y artefactos del dashboard. Única puerta a los archivos.

- Los datos son los del Jupyter Book: el EDA se describe sobre el conjunto
  de entrenamiento y la prueba se reserva para evaluar modelos.
- Todo se lee una vez por proceso (lru_cache) y las funciones devuelven
  copias cuando el llamador podría modificarlas.
- Si falta un artefacto, las funciones ``*_disponible`` lo dicen y la página
  muestra un aviso en lugar de fallar.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C


# ---- Datos procesados ---------------------------------------------------- #
@lru_cache(maxsize=1)
def roles() -> dict:
    return json.loads(C.ROLES.read_text(encoding="utf-8"))


def predictores() -> list[str]:
    r = roles()
    return r["numericas"] + r["ordinales"] + r["binarias"] + r["categoricas"]


def objetivo() -> str:
    return roles()["objetivo"]


@lru_cache(maxsize=1)
def etiquetas_edad() -> dict[int, str]:
    return {int(k): v for k, v in roles()["etiquetas_edad"].items()}


@lru_cache(maxsize=1)
def _entrenamiento() -> pd.DataFrame:
    df = pd.read_parquet(C.TRAIN)
    # Categóricas como category: reduce la memoria a una fracción (importa
    # en el plan gratuito de Render) y acelera los filtros.
    for columna in roles()["categoricas"]:
        df[columna] = df[columna].astype(str).astype("category")
    return df


def entrenamiento() -> pd.DataFrame:
    return _entrenamiento()


@lru_cache(maxsize=1)
def prueba() -> pd.DataFrame:
    return pd.read_parquet(C.PRUEBA)


def es_numerica(variable: str) -> bool:
    r = roles()
    return variable in r["numericas"] + r["ordinales"] + r["binarias"]


# ---- Filtros globales de la página EDA ----------------------------------- #
@lru_cache(maxsize=1)
def opciones_filtro() -> dict:
    df = entrenamiento()
    return {
        "edad": [{"label": etiquetas_edad()[k], "value": k}
                 for k in sorted(df["age_ord"].unique())],
        "sexo": sorted(df["gender"].astype(str).unique()),
        "raza": sorted(df["race"].astype(str).unique()),
        "admision": sorted(df["admission_type_id"].astype(str).unique()),
        "dias": (int(df["time_in_hospital"].min()), int(df["time_in_hospital"].max())),
    }


def filtrar(edades=None, sexos=None, razas=None, admisiones=None, dias=None) -> pd.DataFrame:
    """Entrenamiento con los filtros de la página aplicados (vacío = sin filtro)."""
    df = entrenamiento()
    mascara = np.ones(len(df), dtype=bool)
    for columna, valores in [("age_ord", edades), ("gender", sexos),
                             ("race", razas), ("admission_type_id", admisiones)]:
        if valores:
            mascara &= df[columna].astype(str).isin([str(v) for v in valores]).to_numpy()
    if dias:
        mascara &= df["time_in_hospital"].between(dias[0], dias[1]).to_numpy()
    return df[mascara]


def tasa_por(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Encuentros y tasa de readmisión por nivel de una variable."""
    y = objetivo()
    datos = df.groupby(variable, observed=True)[y].agg(["size", "mean"]).reset_index()
    datos.columns = [variable, "n", "tasa"]
    datos["tasa"] *= 100
    return datos


# ---- Artefactos del modelado --------------------------------------------- #
def artefactos_disponibles() -> bool:
    return (C.ARTEFACTOS / "mejores_combinaciones.csv").exists()


@lru_cache(maxsize=1)
def mejores() -> pd.DataFrame:
    return pd.read_csv(C.ARTEFACTOS / "mejores_combinaciones.csv")


@lru_cache(maxsize=8)
def metadata(modelo: str) -> dict:
    return json.loads((C.ARTEFACTOS / f"{modelo}.metadata.json").read_text(encoding="utf-8"))


def modelos_disponibles() -> list[str]:
    """Modelos con artefacto y metadatos, en el orden de la tabla de líderes."""
    if not artefactos_disponibles():
        return []
    return [m for m in mejores()["modelo"]
            if (C.ARTEFACTOS / f"{m}.joblib").exists()
            and (C.ARTEFACTOS / f"{m}.metadata.json").exists()]


@lru_cache(maxsize=1)
def predicciones_prueba() -> pd.DataFrame | None:
    ruta = C.ARTEFACTOS / "predicciones_prueba.parquet"
    return pd.read_parquet(ruta) if ruta.exists() else None


@lru_cache(maxsize=1)
def shap_global() -> pd.DataFrame | None:
    ruta = C.ARTEFACTOS / "shap_global.parquet"
    return pd.read_parquet(ruta) if ruta.exists() else None


@lru_cache(maxsize=1)
def faltantes() -> pd.DataFrame | None:
    ruta = C.ARTEFACTOS / "eda_faltantes.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


@lru_cache(maxsize=1)
def por_fold_mejores() -> pd.DataFrame:
    """AUC-PR por fold externo de la mejor combinación de cada modelo.

    Es la matriz de la comparación de Friedman y Nemenyi del capítulo 11. Si
    existe el resultado del modelo original con las mismas particiones
    (artifacts/modelo_original/por_fold.json), se añade como una columna más.
    """
    maestra = pd.read_csv(C.TABLA_MAESTRA).query("estado == 'completada'")
    elegidas = maestra.loc[maestra.groupby("modelo")["auc_pr_media"].idxmax()]
    matriz = pd.DataFrame({f["modelo"]: json.loads(f["auc_pr_por_fold"])
                           for _, f in elegidas.iterrows()})
    ruta = C.ORIGINAL / "por_fold.json"
    if ruta.exists():
        original = json.loads(ruta.read_text(encoding="utf-8"))
        if len(original["auc_pr_por_fold"]) == len(matriz):
            matriz[original.get("nombre", "Modelo original")] = original["auc_pr_por_fold"]
    return matriz


@lru_cache(maxsize=1)
def tamanos_efecto() -> pd.DataFrame | None:
    ruta = C.RESULTADOS / "tamanos_efecto_cohen.csv"
    return pd.read_csv(ruta, index_col=0) if ruta.exists() else None


def archivo_original(nombre: str):
    """Contenido de un archivo del modelo original, o None si aún no existe."""
    ruta = C.ORIGINAL / nombre
    if not ruta.exists():
        return None
    texto = ruta.read_text(encoding="utf-8")
    return json.loads(texto) if ruta.suffix == ".json" else texto
