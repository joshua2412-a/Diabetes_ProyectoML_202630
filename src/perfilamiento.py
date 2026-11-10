"""
Medición de tiempo, memoria y desempeño de un ajuste.

Forma parte de ``src/``, la versión modular del motor del experimento. Las
funciones se trasladaron sin cambios de lógica desde ``docs/experimento.py``,
que se conserva congelado para reproducir el Entregable 2; la equivalencia
entre ambos se verifica en ``tests/test_equivalencia_e2.py``.
"""

from __future__ import annotations

import time
import tracemalloc

from sklearn.base import clone
from sklearn.metrics import average_precision_score


def medir_ajuste(pipeline, X_ent, y_ent, X_eval=None, y_eval=None,
                 medir_memoria=True):
    """Mide tiempo, memoria pico y desempeño de un ajuste.

    El tiempo se mide **sin** ``tracemalloc``: el perfilador de memoria
    registra cada asignación y ralentiza de forma desigual las operaciones
    con muchas asignaciones pequeñas (preprocesamiento, matrices dispersas),
    lo que añadía un piso común de decenas de segundos a todos los modelos y
    aplastaba las diferencias que se quieren medir. La memoria pico se mide
    en un segundo ajuste independiente, sobre una copia sin ajustar.

    ``tracemalloc`` registra las asignaciones de Python y de numpy en el
    proceso principal; no ve la memoria interna de bibliotecas en C++
    (XGBoost) ni la de procesos hijos, así que en esos casos es una cota
    inferior.

    Parameters
    ----------
    medir_memoria : bool, default True
        Si es ``False`` se omite el segundo ajuste (la mitad del costo).

    Returns
    -------
    dict
        ``ajuste_s``, ``memoria_pico_mb`` (si se mide) y, si se pasa un
        conjunto de evaluación, ``inferencia_s`` y ``auc_pr``.
    """
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    resultado = {"ajuste_s": time.perf_counter() - inicio}

    if medir_memoria:
        copia = clone(pipeline)
        tracemalloc.start()
        copia.fit(X_ent, y_ent)
        _, pico = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        resultado["memoria_pico_mb"] = pico / 2**20

    if X_eval is not None:
        inicio = time.perf_counter()
        probabilidades = pipeline.predict_proba(X_eval)[:, 1]
        resultado["inferencia_s"] = time.perf_counter() - inicio
        resultado["auc_pr"] = average_precision_score(y_eval, probabilidades)
    return resultado
