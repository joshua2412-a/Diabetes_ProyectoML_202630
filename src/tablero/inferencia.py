"""
Predicción en vivo y su explicación local.

Carga los pipelines serializados por experiments/entrenar_artefactos.py. Se
cargan bajo demanda y como mucho dos a la vez: el plan gratuito de Render
tiene poca memoria y un Random Forest profundo puede ocupar cientos de MB.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.explicacion import shapley_permutacion

from . import config as C
from . import datos as D


@lru_cache(maxsize=2)
def cargar(modelo: str):
    return joblib.load(C.ARTEFACTOS / f"{modelo}.joblib")


@lru_cache(maxsize=1)
def fondo() -> pd.DataFrame:
    return pd.read_parquet(C.ARTEFACTOS / "fondo_shap.parquet")


def esquema() -> tuple[dict, list[str]]:
    """Esquema de entrada común a todos los modelos (el del primero disponible)."""
    meta = D.metadata(D.modelos_disponibles()[0])
    return meta["esquema"], meta["orden_columnas"]


#: Variables que el ETL deriva de otras (capítulo 2). El formulario no las
#: pide: se calculan aquí con la misma regla, para que el caso no pueda ser
#: incoherente (por ejemplo, 3 hospitalizaciones previas y «Hospitalización
#: previa = No»).
DERIVADAS = ("hospitalizacion_previa", "n_farmacos_activos")


def derivar(fila: dict) -> dict:
    """Añade las variables derivadas con la regla del ETL."""
    fila = dict(fila)
    fila["hospitalizacion_previa"] = int(fila["number_inpatient"] > 0)
    fila["n_farmacos_activos"] = sum(fila[f] in ("Up", "Down", "Steady")
                                     for f in D.roles()["farmacos"])
    return fila


def construir_caso(valores: dict) -> tuple[pd.DataFrame | None, list[str]]:
    """Valida lo que escribió el usuario y arma la fila de entrada.

    Returns
    -------
    tuple
        (caso, errores). Si hay errores, ``caso`` es None y cada error
        nombra la variable y el rango válido.
    """
    reglas, orden = esquema()
    errores, fila = [], {}
    for columna in orden:
        if columna in DERIVADAS:
            continue
        regla, valor = reglas[columna], valores.get(columna)
        if valor is None or valor == "":
            errores.append(f"{C.nombre(columna)}: falta el valor.")
            continue
        if regla["tipo"] == "numerica":
            try:
                numero = float(valor)
            except (TypeError, ValueError):
                errores.append(f"{C.nombre(columna)}: debe ser un número.")
                continue
            if not regla["min"] <= numero <= regla["max"]:
                errores.append(f"{C.nombre(columna)}: debe estar entre "
                               f"{regla['min']:g} y {regla['max']:g}.")
                continue
            if regla["entero"] and not float(numero).is_integer():
                errores.append(f"{C.nombre(columna)}: debe ser un número entero.")
                continue
            fila[columna] = int(numero) if regla["entero"] else numero
        else:
            if str(valor) not in regla["niveles"]:
                errores.append(f"{C.nombre(columna)}: «{valor}» no es un nivel válido.")
                continue
            fila[columna] = str(valor)
    if errores:
        return None, errores
    caso = pd.DataFrame([derivar(fila)], columns=orden)
    # Mismos tipos que el conjunto de fondo, para que el pipeline vea lo mismo
    # que vio al entrenar.
    referencia = fondo()
    for columna in orden:
        if pd.api.types.is_numeric_dtype(referencia[columna]):
            caso[columna] = caso[columna].astype(referencia[columna].dtype)
    return caso, []


def predecir(modelo: str, caso: pd.DataFrame) -> float:
    return float(cargar(modelo).predict_proba(caso)[:, 1][0])


def explicar(modelo: str, caso: pd.DataFrame) -> dict:
    """Valores de Shapley del caso sobre las variables originales."""
    pipeline = cargar(modelo)
    return shapley_permutacion(lambda t: pipeline.predict_proba(t)[:, 1], caso,
                               fondo()[caso.columns], C.PERMUTACIONES_EN_VIVO)


def percentil_en_prueba(modelo: str, p: float) -> float | None:
    """Qué porcentaje de los pacientes de prueba tiene un riesgo menor."""
    predicciones = D.predicciones_prueba()
    if predicciones is None or modelo not in predicciones:
        return None
    return float(100 * np.mean(predicciones[modelo].to_numpy() < p))
