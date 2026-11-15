"""
Predicción en vivo y su explicación local.

Carga los pipelines serializados por experiments/entrenar_artefactos.py. Se
cargan bajo demanda y como mucho dos a la vez: el plan gratuito de Render
tiene poca memoria y un Random Forest profundo puede ocupar cientos de MB.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.explicacion import shapley_permutacion

from . import config as C
from . import datos as D


#: Como mucho dos pipelines en memoria a la vez (los últimos usados). Es
#: una caché LRU explícita en lugar de ``lru_cache`` porque el consenso de
#: los 7 modelos necesita saber cuáles están cargados sin cargar ninguno.
_MAXIMO_EN_MEMORIA = 2
_CACHE: OrderedDict = OrderedDict()
_CANDADO = threading.Lock()   # gunicorn atiende con varios hilos


def cargar(modelo: str):
    with _CANDADO:
        if modelo in _CACHE:
            _CACHE.move_to_end(modelo)
            return _CACHE[modelo]
    pipeline = joblib.load(C.ARTEFACTOS / f"{modelo}.joblib")
    with _CANDADO:
        _CACHE[modelo] = pipeline
        _CACHE.move_to_end(modelo)
        while len(_CACHE) > _MAXIMO_EN_MEMORIA:
            _CACHE.popitem(last=False)
    return pipeline


def en_memoria(modelo: str) -> bool:
    with _CANDADO:
        return modelo in _CACHE


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


def predecir_todos(caso: pd.DataFrame) -> pd.DataFrame:
    """Probabilidad del caso según cada uno de los modelos disponibles.

    No pasa por la caché de ``cargar`` para los modelos que no están ya en
    ella: así consultar a los 7 no desaloja los dos que el usuario está
    usando, y el modelo de paso se libera al terminar (importa con los
    512 MB del plan gratuito de Render). Cuesta unos segundos más la primera
    vez, a cambio de no subir el pico de memoria.
    """
    filas = []
    for modelo in D.modelos_disponibles():
        pipeline = (cargar(modelo) if en_memoria(modelo)
                    else joblib.load(C.ARTEFACTOS / f"{modelo}.joblib"))
        filas.append({"modelo": modelo, "p": float(pipeline.predict_proba(caso)[:, 1][0]),
                      "umbral": float(D.metadata(modelo)["umbral"])})
        del pipeline
    return pd.DataFrame(filas)


#: Variables del «¿qué pasaría si...?»: las que se pueden mover en el
#: formulario con un sentido clínico claro, y el tope del barrido (p99 del
#: entrenamiento aproximado: más allá casi no hay pacientes).
QUE_PASA_SI = {"time_in_hospital": 14, "num_medications": 45, "number_inpatient": 8,
               "number_emergency": 6, "number_diagnoses": 16, "num_lab_procedures": 90,
               "num_procedures": 6, "number_outpatient": 8}


def barrido(modelo: str, caso: pd.DataFrame, variable: str) -> tuple[list, np.ndarray]:
    """Probabilidad del caso al mover una sola variable, con el resto fijo.

    Las variables derivadas se recalculan en cada valor con la regla del ETL
    (si se mueven las hospitalizaciones previas, también cambia
    «hospitalización previa»). Es una sola llamada a ``predict_proba``.
    """
    reglas, _ = esquema()
    minimo = int(reglas[variable]["min"])
    maximo = int(min(reglas[variable]["max"], QUE_PASA_SI.get(variable, reglas[variable]["max"])))
    paso = max(1, int(np.ceil((maximo - minimo) / 24)))
    valores = list(range(minimo, maximo + 1, paso))
    actual = int(caso[variable].iloc[0])
    if actual not in valores and minimo <= actual <= reglas[variable]["max"]:
        valores = sorted(set(valores) | {actual})
    base = caso.iloc[0].to_dict()
    filas = []
    for v in valores:
        fila = dict(base)
        fila[variable] = v
        filas.append(derivar(fila))
    tabla = pd.DataFrame(filas, columns=caso.columns)
    for columna in caso.columns:
        tabla[columna] = tabla[columna].astype(caso[columna].dtype)
    return valores, cargar(modelo).predict_proba(tabla)[:, 1]
