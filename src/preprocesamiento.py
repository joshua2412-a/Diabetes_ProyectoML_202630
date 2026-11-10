"""
Preprocesador por bloques de variables, ajustable dentro de cada fold.

Forma parte de ``src/``, la versión modular del motor del experimento. Las
funciones se trasladaron sin cambios de lógica desde ``docs/experimento.py``,
que se conserva congelado para reproducir el Entregable 2; la equivalencia
entre ambos se verifica en ``tests/test_equivalencia_e2.py``.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def construir_preprocesador(roles, min_frecuencia=0.01, salida_densa=False):
    """Ensambla el preprocesador como transformador ajustable dentro de folds.

    Cada bloque de variables recibe el tratamiento que su naturaleza exige:

    * Numéricas y ordinales: imputación por mediana y escalado robusto
      (mediana y rango intercuartílico). El capítulo 3 rechazó la normalidad
      en todas las variables numéricas y documentó atípicos clínicamente
      plausibles, situación en la que la estandarización clásica queda
      distorsionada por las colas.
    * Binarias: sin transformación. Ya están en {0, 1}.
    * Categóricas: imputación por moda y codificación one-hot. Las categorías
      con frecuencia inferior al umbral se agrupan, lo que evita columnas casi
      vacías cuando un nivel poco frecuente falta en el fold de validación.
      ``race`` se codifica aparte y sin agrupar, para conservar desagregado
      el grupo minoritario que exige el análisis de equidad.

    Todos los estimadores del preprocesador (mediana, IQR, moda, niveles
    observados) se ajustan con los datos del fold de entrenamiento
    exclusivamente, que es la condición para que no haya fuga.

    Parameters
    ----------
    roles : dict
        Diccionario de roles exportado por el capítulo 2.
    min_frecuencia : float, default 0.01
        Frecuencia relativa mínima para que una categoría reciba columna
        propia en la codificación one-hot.
    salida_densa : bool, default False
        Si es ``True``, la salida es una matriz densa. Se usa en los modelos
        de :data:`REQUIERE_DENSO`.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocesador sin ajustar.
    """
    numericas = roles["numericas"] + roles["ordinales"]
    binarias = roles["binarias"]
    # race va en un bloque sin agrupación por frecuencia: el capítulo 2 la
    # exime del umbral del 1 % para conservar desagregado el grupo Asian,
    # y min_frequency la volvería a agrupar dentro de cada fold.
    categoricas = [c for c in roles["categoricas"] if c != "race"]

    bloque_numerico = Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", RobustScaler()),
    ])
    bloque_categorico = Pipeline([
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("codificacion", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=min_frecuencia,
            sparse_output=not salida_densa,
        )),
    ])

    bloques = [
        ("num", bloque_numerico, numericas),
        ("bin", "passthrough", binarias),
    ]
    if "race" in roles["categoricas"]:
        bloques.append(("race", OneHotEncoder(
            handle_unknown="ignore", sparse_output=not salida_densa),
            ["race"]))
    bloques.append(("cat", bloque_categorico, categoricas))

    # sparse_threshold=0 con salida densa fuerza un ndarray; con salida
    # dispersa, el umbral por defecto (0.3) conserva la matriz dispersa
    # porque su densidad ronda el 22 %.
    return ColumnTransformer(
        bloques, verbose_feature_names_out=False,
        sparse_threshold=0.0 if salida_densa else 0.3,
    )


def nombres_variables(preprocesador):
    """Nombres de las columnas que produce un preprocesador ya ajustado."""
    return list(preprocesador.get_feature_names_out())
