"""Pruebas de los optimizadores (src/optimizacion.py)."""

import numpy as np
import pytest
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

from src.balanceo import construir_pipeline
from src.espacios import espacio_busqueda, rejilla_hiperparametros
from src.optimizacion import optimizar


@pytest.fixture
def busqueda(datos, roles):
    """Argumentos comunes de una búsqueda con la regresión logística."""
    X, y, grupos = datos
    particiones = list(StratifiedGroupKFold(n_splits=3, shuffle=True,
                                            random_state=42)
                       .split(X, y, groups=grupos))
    return dict(pipeline=construir_pipeline("logistica", "ninguno", roles),
                rejilla=rejilla_hiperparametros("logistica"),
                espacio=espacio_busqueda("logistica"),
                X=X, y=y, particiones=particiones,
                metrica="average_precision")


def _parametros(resultado):
    """Configuraciones evaluadas, sin sus puntajes."""
    return [{k: v for k, v in e.items()
             if k not in {"puntaje", "folds_evaluados"}}
            for e in resultado["evaluaciones"]]


@pytest.mark.parametrize("estrategia", ["random", "optuna", "deap"])
def test_respeta_el_presupuesto(busqueda, estrategia):
    resultado = optimizar(estrategia, presupuesto=6, **busqueda)
    assert len(resultado["evaluaciones"]) == 6


def test_grid_recorre_la_rejilla_completa(busqueda):
    resultado = optimizar("grid", presupuesto=3, **busqueda)
    assert len(resultado["evaluaciones"]) == len(
        ParameterGrid(busqueda["rejilla"]))


@pytest.mark.parametrize("estrategia", ["grid", "random", "optuna", "deap"])
def test_curva_anytime_nunca_empeora(busqueda, estrategia):
    curva = [v for v in optimizar(estrategia, presupuesto=6,
                                  **busqueda)["historial"] if v is not None]
    assert all(b >= a for a, b in zip(curva, curva[1:]))


@pytest.mark.parametrize("estrategia", ["grid", "random", "deap"])
def test_devuelve_la_mejor_evaluada(busqueda, estrategia):
    resultado = optimizar(estrategia, presupuesto=6, **busqueda)
    mejor = max(e["puntaje"] for e in resultado["evaluaciones"])
    assert resultado["mejor_puntaje"] == pytest.approx(mejor)


def test_random_presupuestos_menores_son_prefijos(busqueda):
    """Las k primeras configuraciones no dependen del presupuesto total.

    Es la propiedad en que se apoya el estudio de convergencia del
    capítulo 12: un presupuesto menor es un prefijo de uno mayor.
    """
    corta = optimizar("random", presupuesto=4, semilla=7, **busqueda)
    larga = optimizar("random", presupuesto=10, semilla=7, **busqueda)
    assert _parametros(larga)[:4] == _parametros(corta)
    assert ([e["puntaje"] for e in larga["evaluaciones"][:4]]
            == [e["puntaje"] for e in corta["evaluaciones"]])


def test_paralelizar_no_cambia_los_resultados(busqueda):
    """n_jobs solo cambia el tiempo, no los puntajes."""
    serie = optimizar("random", presupuesto=6, n_jobs=1, **busqueda)
    paralelo = optimizar("random", presupuesto=6, n_jobs=2, **busqueda)
    assert ([e["puntaje"] for e in serie["evaluaciones"]]
            == [e["puntaje"] for e in paralelo["evaluaciones"]])


@pytest.mark.parametrize("estrategia", ["random", "optuna", "deap"])
def test_misma_semilla_misma_busqueda(busqueda, estrategia):
    a = optimizar(estrategia, presupuesto=6, semilla=3, **busqueda)
    b = optimizar(estrategia, presupuesto=6, semilla=3, **busqueda)
    assert a["mejores_parametros"] == b["mejores_parametros"]


def test_estrategia_desconocida(busqueda):
    with pytest.raises(ValueError):
        optimizar("inventada", presupuesto=3, **busqueda)
