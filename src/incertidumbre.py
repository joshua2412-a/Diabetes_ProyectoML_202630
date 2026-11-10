"""
Calidad probabilística y cuantificación de incertidumbre (cap. 13, sección 13.8).

* **Error de calibración esperado (ECE).** Promedio, ponderado por el
  número de casos, de la distancia entre la probabilidad media predicha y
  la frecuencia observada en cada bin.
* **Predicción conforme inductiva por clase (Mondrian).** Con un conjunto de
  calibración que el modelo no usó para ajustarse, el puntaje de no
  conformidad de un caso es ``1 - p(clase real)``. Para cada clase ``c`` se
  toma el cuantil ``ceil((n_c + 1)(1 - alfa)) / n_c`` de los puntajes de los
  casos de esa clase, y el conjunto de predicción de un caso nuevo contiene
  las clases cuyo puntaje no supera su cuantil. Bajo intercambiabilidad, la
  cobertura de cada clase es al menos ``1 - alfa`` (Vovk et al., 2005); el
  condicionamiento por clase evita que la cobertura se cumpla solo en la
  clase mayoritaria, que con un 11 % de positivos es el riesgo principal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def error_calibracion(y_real, probabilidades, bins=10, estrategia="cuantiles"):
    """ECE con bins de igual frecuencia (por defecto) o de igual ancho."""
    y = np.asarray(y_real, dtype=float)
    p = np.asarray(probabilidades, dtype=float)
    if estrategia == "cuantiles":
        bordes = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    else:
        bordes = np.linspace(0, 1, bins + 1)
    bin_ = np.clip(np.searchsorted(bordes, p, side="right") - 1, 0, len(bordes) - 2)
    total = 0.0
    for b in np.unique(bin_):
        m = bin_ == b
        total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / len(y))


def cuantiles_conformes(y_calibracion, p_calibracion, alfa=0.1):
    """Cuantil del puntaje de no conformidad de cada clase (0 y 1)."""
    y = np.asarray(y_calibracion).astype(int)
    p = np.asarray(p_calibracion, dtype=float)
    cuantiles = {}
    for clase in (0, 1):
        puntajes = 1 - (p[y == 1] if clase == 1 else 1 - p[y == 0])
        n = len(puntajes)
        if n == 0:
            raise ValueError(f"La calibración no tiene casos de la clase {clase}")
        nivel = np.ceil((n + 1) * (1 - alfa)) / n
        cuantiles[clase] = (float(np.quantile(puntajes, nivel, method="higher"))
                            if nivel <= 1 else np.inf)
    return cuantiles


def conjuntos_prediccion(p, cuantiles):
    """Matriz booleana (n, 2): si cada clase entra en el conjunto de predicción."""
    p = np.asarray(p, dtype=float)
    # Puntaje de no conformidad: 1 - p(clase). Para la clase 0 es p; para la 1, 1 - p.
    return np.column_stack([p <= cuantiles[0], (1 - p) <= cuantiles[1]])


def evaluar_conformal(y_calibracion, p_calibracion, y_prueba, p_prueba, alfa=0.1):
    """Cobertura empírica y tamaño de los conjuntos en el conjunto de prueba."""
    q = cuantiles_conformes(y_calibracion, p_calibracion, alfa)
    conjuntos = conjuntos_prediccion(p_prueba, q)
    y = np.asarray(y_prueba).astype(int)
    cubre = conjuntos[np.arange(len(y)), y]
    tamano = conjuntos.sum(axis=1)
    filas = [{"grupo": "Todos", "n": len(y), "cobertura": cubre.mean()}]
    for clase, nombre in ((0, "No readmitidos"), (1, "Readmitidos")):
        m = y == clase
        filas.append({"grupo": nombre, "n": int(m.sum()), "cobertura": cubre[m].mean()})
    tabla = pd.DataFrame(filas)
    tabla["cobertura nominal"] = 1 - alfa
    resumen = {"cuantil_clase_0": q[0], "cuantil_clase_1": q[1],
               "tamano_medio": float(tamano.mean()),
               "conjuntos_vacios": float((tamano == 0).mean()),
               "conjuntos_con_ambas": float((tamano == 2).mean()),
               "conjuntos_solo_readmitido": float((conjuntos[:, 1] & ~conjuntos[:, 0]).mean())}
    return tabla, resumen
