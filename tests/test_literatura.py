"""Pruebas de la lectura de resultados de Google Scholar copiados a texto."""

import os
import time

import pandas as pd

from src.literatura import leer_google_scholar

PAGINA_1 = """
"dynamic ensemble selection" readmission
Usuario

Artículos
Aproximadamente 39 resultados (0,08 s)
Cualquier momento
[HTML] jmir.org
Lectura rápida
[HTML] Predicting Unplanned Readmission Risk in Patients With Cirrhosis
Z Shi, L Huang, X Xu… - JMIR Medical …, 2025 - medinform.jmir.org
… the risk of readmission … META-DES …
Guardar Citar Artículos relacionados Las 9 versiones
Ensemble pruning with Pareto search
R Hu, S Zhou - Computational intelligence and …, 2019 - Wiley Online Library
… dynamic ensemble selection …
Guardar Citar Citado por 4 Artículos relacionados
[HTML] acm.org
Lectura rápida - arxiv.org
“A 6 or a 9?”: Ensemble Learning through the Multiplicity of Performant Models
G Zuin, A Veloso - ACM Transactions on Knowledge Discovery from Data, 2025 - dl.acm.org
… Diabetes Readmission …
Guardar Citar Citado por 1
"""

PAGINA_2 = """
Página 2 de 29 resultados (0,07 s)
Dynamic Ensemble Construction via Heuristic Optimization
SY Saglam, WN Street - academia.edu
… sin año …
Guardar Citar
[HTML] Predicting Unplanned Readmission Risk in Patients With Cirrhosis
Z Shi, L Huang, X Xu… - JMIR Medical …, 2025 - medinform.jmir.org
… repetido …
Guardar Citar
"""

SIN_RESULTADOS = """
"selección dinámica" clasificadores readmisión diabetes
Su consulta, "selección dinámica", no ha obtenido ningún artículo como resultado.
"""


def test_lee_registros_total_y_fecha(tmp_path):
    (tmp_path / "C1.txt").write_text(PAGINA_1 + PAGINA_2, encoding="utf-8")
    (tmp_path / "C5.txt").write_text(SIN_RESULTADOS, encoding="utf-8")
    instante = time.mktime((2026, 11, 10, 12, 0, 0, 0, 0, -1))
    os.utime(tmp_path / "C1.txt", (instante, instante))

    busquedas, registros = leer_google_scholar(tmp_path)

    totales = dict(zip(busquedas["cadena"], busquedas["resultados"]))
    assert totales == {"C1": 29, "C5": 0}          # la página 2 corrige la estimación
    assert busquedas.set_index("cadena").loc["C1", "fecha"] == "2026-11-10"
    assert busquedas.set_index("cadena").loc["C1", "consulta"] == '"dynamic ensemble selection" readmission'

    c1 = registros.query("cadena == 'C1'").set_index("título")
    assert len(c1) == 4                             # el repetido cuenta una vez
    shi = c1.loc["Predicting Unplanned Readmission Risk in Patients With Cirrhosis"]
    assert shi["año"] == 2025 and shi["enlace"] == "medinform.jmir.org"
    # La línea «Lectura rápida - dominio» no se confunde con la de autores.
    assert "“A 6 or a 9?”: Ensemble Learning through the Multiplicity of Performant Models" in c1.index
    # Fuente con espacios y registro sin año.
    assert c1.loc["Ensemble pruning with Pareto search", "enlace"] == "Wiley Online Library"
    assert pd.isna(c1.loc["Dynamic Ensemble Construction via Heuristic Optimization", "año"])
