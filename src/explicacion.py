"""
Explicación local con valores de Shapley, sobre las variables originales.

El dashboard explica cada predicción en términos de las 38 variables que
describe el formulario (edad, hospitalizaciones previas, destino al alta...)
y no de las ~100 columnas que produce la codificación one-hot. Por eso se
estiman los valores de Shapley directamente sobre el pipeline completo, que
recibe la tabla original, en lugar de explicar solo el modelo final.

Método: muestreo de permutaciones (Štrumbelj y Kononenko, 2014) con la
definición intervencional del valor de una coalición: las variables fuera de
la coalición toman los valores de un conjunto de fondo. Para cada permutación
se recorren las variables en su orden y la contribución de cada una es el
cambio medio en la predicción al fijarla en el valor del caso. Se usan
permutaciones antitéticas (cada una con su inversa), que reducen la varianza
del estimador sin costo adicional.

Propiedades que se cumplen por construcción (y se verifican en
tests/test_explicacion.py):

* Eficiencia exacta: las contribuciones suman f(caso) - E[f(fondo)] en
  cualquier número de permutaciones, porque cada recorrido es una suma
  telescópica.
* Variable muda: una variable que el modelo no usa recibe exactamente 0.
* Linealidad: en un modelo lineal, la contribución de cada variable es
  exactamente beta_j * (x_j - media del fondo), sin error de muestreo.

Costo: todas las coaliciones se evalúan en UNA llamada al modelo, con
``n_permutaciones * (p + 1) * len(fondo)`` filas. Con 8 permutaciones, 38
variables y 20 filas de fondo son 6,240 filas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .configuracion import SEMILLA


def _permutaciones(p, n, aleatorio):
    """``n`` permutaciones antitéticas de ``p`` índices (n se redondea a par)."""
    salida = []
    while len(salida) < n:
        orden = aleatorio.permutation(p)
        salida.extend([orden, orden[::-1]])
    return salida[:max(2, n + n % 2)]


def shapley_permutacion(funcion, caso, fondo, n_permutaciones=8,
                        semilla=SEMILLA):
    """Valores de Shapley de una predicción por muestreo de permutaciones.

    Parameters
    ----------
    funcion : callable
        Recibe un DataFrame con las columnas de ``fondo`` y devuelve un
        arreglo con una predicción por fila (por ejemplo, la probabilidad de
        la clase positiva).
    caso : pandas.DataFrame or pandas.Series
        El caso que se explica (una fila).
    fondo : pandas.DataFrame
        Conjunto de referencia para las variables ausentes de una coalición.
    n_permutaciones : int, default 8
    semilla : int

    Returns
    -------
    dict
        ``contribuciones`` (Series indexada por variable), ``base`` (media
        de la predicción sobre el fondo) y ``prediccion`` (la del caso).
    """
    fila = caso.iloc[0] if isinstance(caso, pd.DataFrame) else caso
    columnas = list(fondo.columns)
    p, b = len(columnas), len(fondo)
    aleatorio = np.random.default_rng(semilla)
    ordenes = _permutaciones(p, n_permutaciones, aleatorio)
    m = len(ordenes)

    # posicion[i, j]: lugar que ocupa la variable j en la permutación i.
    posicion = np.empty((m, p), dtype=int)
    for i, orden in enumerate(ordenes):
        posicion[i, orden] = np.arange(p)
    # fijada[i, k, j]: en el paso k de la permutación i, ¿la variable j ya
    # tomó el valor del caso? En el paso 0 ninguna; en el paso p, todas.
    fijada = posicion[:, None, :] < np.arange(p + 1)[None, :, None]
    fijada = np.repeat(fijada.reshape(m * (p + 1), p), b, axis=0)

    repeticiones = m * (p + 1)
    datos = {}
    for j, columna in enumerate(columnas):
        base = np.tile(fondo[columna].to_numpy(), repeticiones)
        valores = np.where(fijada[:, j], fila[columna], base)
        if pd.api.types.is_numeric_dtype(fondo[columna]):
            valores = valores.astype(fondo[columna].dtype)
        datos[columna] = valores
    tabla = pd.DataFrame(datos, columns=columnas)

    # Media sobre el fondo de cada coalición: v[i, k].
    v = np.asarray(funcion(tabla), dtype=float).reshape(m, p + 1, b).mean(axis=2)
    pasos = np.diff(v, axis=1)
    contribuciones = np.zeros(p)
    for i, orden in enumerate(ordenes):
        contribuciones[orden] += pasos[i]
    contribuciones /= m

    return {"contribuciones": pd.Series(contribuciones, index=columnas),
            "base": float(v[:, 0].mean()),
            "prediccion": float(v[:, -1].mean())}


def shapley_muestra(funcion, casos, fondo, n_permutaciones=8, semilla=SEMILLA):
    """Valores de Shapley de varias filas, para la importancia global.

    Returns
    -------
    pandas.DataFrame
        Una fila por caso y una columna por variable.
    """
    filas = [shapley_permutacion(funcion, casos.iloc[[i]], fondo,
                                 n_permutaciones, semilla + i)["contribuciones"]
             for i in range(len(casos))]
    return pd.DataFrame(filas, index=casos.index)
