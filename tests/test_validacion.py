"""Pruebas de la validación anidada y del ejecutor (src/validacion.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from src import validacion
from src.metricas import METRICAS
from src.validacion import Corrida, ejecutar_corrida, ejecutar_experimento


def _resultado_vacio():
    """Lo mínimo que devuelve un fold, para espiar las particiones."""
    return {"metricas": {m: 0.5 for m in METRICAS}, "mejores_parametros": {},
            "historial": [], "diversidad": [], "n_evaluaciones": 1,
            "n_ajustes": 1, "t_busqueda": 0.0, "t_ajuste": 0.0,
            "t_inferencia": 0.0, "t_fold": 0.0}


def test_particiones_externas_sin_pacientes_compartidos(datos, roles,
                                                        monkeypatch):
    X, y, grupos = datos
    vistas = []

    def espia(i, indices_train, indices_val, **_):
        vistas.append((indices_train, indices_val))
        return _resultado_vacio()

    monkeypatch.setattr(validacion, "_procesar_fold", espia)
    ejecutar_corrida(Corrida("logistica", "ninguno", "random", presupuesto=2),
                     X, y, grupos, roles, folds_externos=5)

    assert len(vistas) == 5
    evaluados = np.concatenate([val for _, val in vistas])
    # Cada encuentro se evalúa exactamente una vez.
    assert sorted(evaluados) == list(range(len(X)))
    for train, val in vistas:
        assert not set(grupos.iloc[train]) & set(grupos.iloc[val])
        # Estratificación: la tasa positiva no se aleja de la global.
        assert abs(y.iloc[val].mean() - y.mean()) < 0.08


@pytest.mark.parametrize("fraccion", [1.0, 0.5])
def test_particiones_internas_y_submuestra_por_paciente(datos, roles,
                                                        monkeypatch, fraccion):
    X, y, grupos = datos
    vistas = []

    def espia(estrategia, pipeline, rejilla, espacio, X_b, y_b, particiones,
              *args, **kwargs):
        vistas.append((X_b.index, particiones))
        return {"mejores_parametros": {}, "mejor_puntaje": 0.0,
                "historial": [], "evaluaciones": [{}], "diversidad": [],
                "ajustes": 0}

    monkeypatch.setattr(validacion, "optimizar", espia)
    ejecutar_corrida(Corrida("logistica", "ninguno", "random", presupuesto=2),
                     X, y, grupos, roles, folds_externos=3, folds_internos=3,
                     fraccion_busqueda=fraccion)

    for indice, particiones in vistas:
        g = grupos.loc[indice]
        # La submuestra toma pacientes completos: como el fold externo ya
        # agrupa por paciente, cada paciente incluido aporta todos sus
        # encuentros del conjunto.
        por_paciente = g.value_counts()
        assert (por_paciente
                == grupos.value_counts()[por_paciente.index]).all()
        for train, val in particiones:
            assert not set(g.iloc[train]) & set(g.iloc[val])
    if fraccion < 1.0:
        assert len(vistas[0][0]) < len(X) * (1 - 1 / 3)


def test_corrida_completa_produce_una_fila_valida(datos, roles):
    X, y, grupos = datos
    fila = ejecutar_corrida(Corrida("logistica", "ninguno", "random",
                                    presupuesto=3),
                            X, y, grupos, roles, folds_externos=3,
                            folds_internos=2)
    assert fila["n_evaluaciones"] == 3 * 3
    assert len(json.loads(fila["auc_pr_por_fold"])) == 3
    assert 0 <= fila["auc_pr_media"] <= 1
    assert fila["presupuesto"] == 3


def test_ejecutor_reanuda_y_registra_no_aplicables(datos, roles, tmp_path,
                                                   capsys):
    X, y, grupos = datos
    ruta = tmp_path / "tabla.csv"
    corridas = [Corrida("logistica", "ninguno", "random", presupuesto=2),
                Corrida("knn", "class_weight", "random", presupuesto=2)]
    argumentos = dict(folds_externos=2, folds_internos=2)

    tabla = ejecutar_experimento(corridas, X, y, grupos, roles, ruta,
                                 **argumentos)
    assert list(tabla["estado"]) == ["completada", "no aplicable"]

    # Segunda ejecución: todo está en disco, no se recalcula nada.
    capsys.readouterr()
    tabla = ejecutar_experimento(corridas, X, y, grupos, roles, ruta,
                                 **argumentos)
    assert len(tabla) == 2
    assert capsys.readouterr().out.count("[omitida]") == 2
    assert len(pd.read_csv(ruta)) == 2
