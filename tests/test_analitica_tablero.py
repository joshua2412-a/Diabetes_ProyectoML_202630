"""Pruebas de los cálculos de las vistas clínicas y de decisión del dashboard
(src/tablero/analitica.py). Usan datos sintéticos: no leen parquet ni
artefactos, para correr igual en la integración continua."""

import numpy as np
import pandas as pd
import pytest

from src.tablero import analitica as A


def test_wilson_contiene_la_proporcion_y_no_sale_de_0_1():
    bajo, alto = A.wilson([0, 5, 50], [10, 50, 50])
    assert np.all(bajo >= 0) and np.all(alto <= 1)
    assert bajo[1] < 0.1 < alto[1]
    assert alto[0] > 0  # con 0 de 10 el intervalo no colapsa en cero


def test_personas_de_100_no_borra_los_extremos():
    assert A.personas_de_100(11.39) == 11
    assert A.personas_de_100(0.4) == 1
    assert A.personas_de_100(0.0) == 0
    assert A.personas_de_100(99.6) == 99


@pytest.mark.parametrize("conteos", [
    {"vp": 715, "fn": 1657, "fp": 2300, "vn": 15431},
    {"vp": 1, "fn": 3, "fp": 2, "vn": 19994},
    {"vp": 0, "fn": 10, "fp": 0, "vn": 90},
])
def test_a_cien_suma_100_y_no_deja_en_cero_un_grupo_con_casos(conteos):
    cien = A.a_cien(conteos)
    assert sum(cien.values()) == 100
    for clave, valor in conteos.items():
        assert (cien[clave] > 0) == (valor > 0)


def test_edad_legible():
    assert A.edad_legible(2) == "20–29"
    assert A.edad_legible(9) == "90–99"


def test_conjunto_conforme_por_clase():
    # p <= q0 incluye «no readmitido»; 1 - p <= q1 incluye «readmitido».
    c0, c1 = A._conjunto(np.array([0.05, 0.5, 0.95]), q0=0.3, q1=0.3)
    assert c0.tolist() == [True, False, False]
    assert c1.tolist() == [False, False, True]


def test_perfil_en_porcentaje():
    df = pd.DataFrame({
        A.Y: [1, 0, 0, 0], "number_inpatient": [2, 0, 0, 1], "number_emergency": [0, 0, 1, 0],
        "admission_type_id": ["Emergency", "Elective", "Emergency", "Urgent"],
        "time_in_hospital": [8, 2, 3, 7], "num_medications": [25, 10, 5, 30],
        "insulin": ["Up", "No", "Steady", "No"], "change": ["Ch", "No", "No", "Ch"],
        "A1Cresult": ["No medida", ">8", "Norm", "No medida"],
        "discharge_disposition_id": ["Discharged to home"] * 2 + ["NULL"] * 2,
    })
    perfil = A.perfil(df)
    assert perfil["Readmitidos en < 30 días"] == 25
    assert perfil["Hospitalización el año previo"] == 50
    assert perfil["Alta a casa"] == 50
    assert perfil.between(0, 100).all()
