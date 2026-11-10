"""
Datos y configuración compartidos por las pruebas.

Las pruebas usan un conjunto sintético pequeño con la misma estructura que
el real (pacientes con varios encuentros, bloques numérico, ordinal, binario
y categórico, y ``race`` aparte), de modo que corren en segundos y no
dependen de los archivos de datos. Solo las pruebas marcadas como ``lento``
leen los datos reales.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import configuracion as cfg

RAIZ = Path(__file__).resolve().parents[1]

ROLES = {
    "objetivo": "y",
    "identificador": "paciente",
    "numericas": ["num_a", "num_b"],
    "ordinales": ["ord"],
    "binarias": ["bin"],
    "categoricas": ["cat_a", "cat_b", "race"],
}
PREDICTORES = (ROLES["numericas"] + ROLES["ordinales"] + ROLES["binarias"]
               + ROLES["categoricas"])


def datos_sinteticos(n_pacientes=300, semilla=0):
    """Encuentros agrupados por paciente con señal conocida.

    Cada paciente tiene de 1 a 3 encuentros y un riesgo latente común, que
    hace que sus encuentros se parezcan entre sí, como en los datos reales.
    El desenlace depende de ``num_a``, ``bin`` y del nivel ``D`` de
    ``cat_a``.
    """
    aleatorio = np.random.default_rng(semilla)
    filas = []
    for paciente in range(n_pacientes):
        riesgo = aleatorio.normal()
        for _ in range(int(aleatorio.integers(1, 4))):
            num_a = aleatorio.normal(riesgo, 1.0)
            binaria = int(aleatorio.integers(0, 2))
            cat_a = str(aleatorio.choice(["A", "B", "C", "D"],
                                         p=[0.5, 0.3, 0.15, 0.05]))
            logit = -2.0 + 1.2 * num_a + 0.8 * binaria + 0.7 * (cat_a == "D")
            filas.append({
                "paciente": paciente,
                "num_a": num_a,
                "num_b": aleatorio.exponential(1.0),
                "ord": int(aleatorio.integers(0, 9)),
                "bin": binaria,
                "cat_a": cat_a,
                "cat_b": str(aleatorio.choice(["X", "Y"])),
                "race": str(aleatorio.choice(
                    ["Caucasian", "AfricanAmerican", "Asian", "Other"],
                    p=[0.7, 0.2, 0.05, 0.05])),
                "y": int(aleatorio.random() < 1 / (1 + np.exp(-logit))),
            })
    return pd.DataFrame(filas)


@pytest.fixture(autouse=True)
def un_solo_proceso():
    """Las pruebas corren en un proceso salvo que pidan otra cosa.

    Evita lanzar procesos en cada prueba (lento en integración continua) y
    restaura el valor al terminar, para que ninguna prueba afecte a otra.
    """
    anterior = cfg.N_JOBS
    cfg.N_JOBS = 1
    yield
    cfg.N_JOBS = anterior


@pytest.fixture
def roles():
    return ROLES


@pytest.fixture
def datos():
    """``(X, y, grupos)`` del conjunto sintético."""
    tabla = datos_sinteticos()
    return (tabla[PREDICTORES], tabla[ROLES["objetivo"]],
            tabla[ROLES["identificador"]])


@pytest.fixture
def razon(datos):
    """Razón entre clases, como la que usa XGBoost para class_weight."""
    _, y, _ = datos
    return float((1 - y.mean()) / y.mean())
