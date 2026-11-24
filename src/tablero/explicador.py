"""
Explicador de DES-CPL para un paciente de prueba (paso 2 de /modelo-original).

Separa el cálculo caro del barato:

- ``caso(indice)`` hace lo caro una sola vez por paciente: las probabilidades
  de los modelos combinados, los ``k`` vecinos positivos y los ``k`` negativos
  de su región de competencia, y la competencia local y la global de cada
  modelo. Se guarda en caché (``lru_cache``) y se devuelve como tipos de
  Python, listos para un ``dcc.Store``.
- ``combinacion(datos, lam, tau)`` hace lo barato: con la competencia ya
  calculada, recombina ``C = (1 - λ) C_local + λ C_global``, los pesos softmax
  y la probabilidad final. Es lo que corre al mover los deslizadores, sin
  volver a buscar vecinos.

Usa el artefacto ``artifacts/des_cpl.joblib`` y las funciones de
``src/des_cpl.py`` (``competencias_locales``, ``competencia_global``,
``pesos_softmax`` y ``combinar``): no reimplementa el mecanismo. La
proyección 2D del espacio de perfiles se calcula una sola vez y se guarda en
caché; solo se dibuja una muestra de los 79.473 pacientes de selección.

Todo es perezoso: nada se carga hasta que se pide el primer paciente.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src import des_cpl as dc

from . import config as C
from . import datos as D
from . import inferencia as I

Y = "readmitted_30"
#: Variables de la ficha del paciente, en el orden en que se muestran.
FICHA = ["age_ord", "time_in_hospital", "number_inpatient", "number_emergency", "num_medications",
         "diag_1_grupo", "A1Cresult"]


def _modelo():
    return I.cargar(C.MODELO_ORIGINAL)


def parametros_artefacto() -> dict:
    """k, τ, λ y modelos combinados del artefacto (los que se marcan como referencia)."""
    m = _modelo()
    return {"k": int(m.k), "tau": float(m.tau), "lam": float(m.encogimiento),
            "modelos": [C.nombre_modelo(n) for n in np.asarray(m.nombres_)[m.activos_]]}


@lru_cache(maxsize=1)
def proyeccion() -> dict:
    """Los dos primeros componentes principales del espacio de perfiles (logits).

    Se calcula una vez con la matriz de covarianza (M × M, con M = 4): no hace
    falta descomponer los 79.473 × 4 valores. El primer componente se orienta
    para que crezca con el riesgo. Incluye también una muestra fija de
    pacientes de fondo para dibujar la nube.
    """
    m = _modelo()
    perfiles = np.asarray(m.region_.perfiles, dtype=float)
    media = perfiles.mean(axis=0)
    centrado = perfiles - media
    valores, vectores = np.linalg.eigh(np.cov(centrado, rowvar=False))
    orden = np.argsort(valores)[::-1][:2]
    componentes = vectores[:, orden].T                     # (2, M)
    for i in range(2):
        if componentes[i].sum() < 0:                       # sentido estable: más riesgo a la derecha
            componentes[i] = -componentes[i]
    azar = np.random.default_rng(42).choice(len(perfiles), size=min(C.PUNTOS_FONDO_PROYECCION, len(perfiles)),
                                            replace=False)
    return {"media": media, "componentes": componentes,
            "varianza": (valores[orden] / valores.sum()).tolist(),
            "fondo": (centrado[azar] @ componentes.T).tolist(),
            "fondo_y": np.asarray(m.region_.y)[azar].astype(int).tolist()}


def _proyectar(perfiles) -> np.ndarray:
    p = proyeccion()
    return (np.asarray(perfiles, dtype=float) - p["media"]) @ p["componentes"].T


def n_pacientes_prueba() -> int:
    return len(D.prueba())


def indice_al_azar(semilla: int, solo_readmitidos: bool = False) -> int:
    """Posición de un paciente de prueba elegido con ``semilla`` (reproducible)."""
    prueba = D.prueba()
    candidatos = np.flatnonzero(prueba[Y].to_numpy() == 1) if solo_readmitidos else np.arange(len(prueba))
    return int(np.random.default_rng(int(semilla)).choice(candidatos))


def ficha(indice: int) -> list[tuple[str, str]]:
    """Datos del paciente que se muestran (nombre, valor legible)."""
    fila = D.prueba().iloc[indice]
    salida = []
    for columna in FICHA:
        if columna not in fila.index:
            continue
        valor = fila[columna]
        if columna == "age_ord":
            texto = D.etiquetas_edad().get(int(valor), str(valor))
        elif D.es_numerica(columna):
            texto = str(int(valor)) if float(valor).is_integer() else str(valor)
        else:
            texto = C.nivel(columna, str(valor))
        salida.append((C.nombre(columna), texto))
    return salida


@lru_cache(maxsize=32)
def caso(indice: int) -> dict:
    """Todo lo que depende solo del paciente (se calcula una vez por paciente).

    Devuelve tipos de Python (listas y números): sirve tal cual para un
    ``dcc.Store``. ``competencia_local`` es la competencia con λ = 0 y
    ``competencia_global`` la del conjunto de selección completo; el resto se
    mezcla en ``combinacion``.
    """
    modelo = _modelo()
    _, orden = I.esquema()
    fila = D.prueba().iloc[[indice]][orden]
    finales, perfiles = modelo.probabilidades_base(fila)
    region, seleccion, k = modelo.region_, modelo._seleccion(), modelo.k
    vecindario = region.vecindario(perfiles, k)
    c_local, vecinos, q = dc.competencias_locales(
        region, vecindario, seleccion, k, competencia=modelo.competencia,
        estratificado=modelo.estratificado, ponderacion=modelo.ponderacion, encogimiento=0.0)
    c_global = dc.competencia_global(seleccion, region.y, modelo.competencia, modelo.ponderacion)
    vecinos = np.asarray(vecinos)[0]
    y_vecinos = region.y[vecinos]
    return {
        "indice": int(indice),
        "modelos": [C.nombre_modelo(n) for n in np.asarray(modelo.nombres_)[modelo.activos_]],
        "probabilidades": np.asarray(finales)[0].tolist(),
        "competencia_local": np.asarray(c_local)[0].tolist(),
        "competencia_global": np.asarray(c_global)[0].tolist(),
        "peso_positivos": float(np.asarray(q).ravel()[0]),
        "paciente_xy": _proyectar(perfiles)[0].tolist(),
        "vecinos_xy": _proyectar(region.perfiles[vecinos]).tolist(),
        "vecinos_y": y_vecinos.astype(int).tolist(),
        "y_real": int(D.prueba().iloc[indice][Y]),
    }


def combinacion(datos: dict, lam: float, tau: float) -> dict:
    """Competencia, pesos y probabilidad final para un λ y un τ dados.

    Solo recombina lo que ``caso`` ya calculó: ``C = (1 - λ) C_local +
    λ C_global``, pesos ``softmax(-C / τ)`` y ``p = Σ w_m p_m``. Con λ = 0,9 y
    τ = 0,0003 reproduce la probabilidad del artefacto (``predict_proba``).
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("λ debe estar entre 0 y 1")
    c_local = np.asarray(datos["competencia_local"], dtype=float)
    c_global = np.asarray(datos["competencia_global"], dtype=float)
    mezcla = (1 - lam) * c_local + lam * c_global
    pesos = dc.pesos_softmax(mezcla[None], tau)
    p = dc.combinar(pesos, np.asarray(datos["probabilidades"], dtype=float)[None])
    return {"competencia": mezcla.tolist(), "pesos": pesos[0].tolist(), "p": float(p[0])}


def p_artefacto(indice: int) -> float:
    """Probabilidad del artefacto para el paciente (``predict_proba`` del modelo serializado)."""
    _, orden = I.esquema()
    return float(_modelo().predict_proba(D.prueba().iloc[[indice]][orden])[0, 1])
