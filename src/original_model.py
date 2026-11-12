"""
Modelo original del Entregable 3 con el nombre y la interfaz que pide la guía.

La guía (sección 7) pide la clase del modelo original en
``src/original_model.py``. La implementación de DES-CPL vive en
``src/des_cpl.py``, junto con sus funciones (región de competencia,
competencia local y global, poda, pesos); este módulo la expone como
``ModeloOriginal`` con dos comodidades para usarla sin el resto del
proyecto:

- ``random_state`` como nombre de la semilla, la convención de
  scikit-learn que usa el esqueleto de pruebas de la guía;
- modelos base por defecto si no se dan ``estimadores``: regresión
  logística, árbol de decisión, Random Forest y Naive Bayes, cuatro
  familias con sesgos distintos, que es lo que la selección dinámica
  necesita para tener algo que elegir.

En el proyecto se usa ``DESCPL`` con los pipelines del benchmark (que
imputan, escalan y balancean dentro del pliegue); los modelos por defecto
de esta clase no imputan, así que ``ModeloOriginal`` rechaza valores
faltantes con un error claro en lugar de dejar que falle un modelo base.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.des_cpl import DESCPL

__all__ = ["ModeloOriginal", "DESCPL"]


def estimadores_por_defecto(semilla):
    """Cuatro modelos base de familias distintas, con la semilla dada."""
    return [
        ("logistica", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("arbol", DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=semilla)),
        ("bosque", RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=1,
                                          random_state=semilla)),
        ("bayes", GaussianNB()),
    ]


class ModeloOriginal(DESCPL):
    """DES-CPL: selección dinámica de ensambles por competencia probabilística local.

    Parámetros
    ----------
    estimadores : list of (str, estimator) or None
        Modelos base. Con None se usan :func:`estimadores_por_defecto`.
    k, tau, n_pliegues, competencia, estratificado, ponderacion,
    excluir_paciente, encogimiento, n_modelos
        Los de :class:`src.des_cpl.DESCPL` (capítulo 15).
    random_state : int
        Semilla de los pliegues y de los modelos por defecto.
    """

    def __init__(self, estimadores=None, k=20, tau=0.01, n_pliegues=3,
                 competencia="brier", estratificado=True, ponderacion="prevalencia",
                 excluir_paciente=True, encogimiento=0.0, n_modelos=None, random_state=42):
        super().__init__(estimadores=estimadores, k=k, tau=tau, n_pliegues=n_pliegues,
                         competencia=competencia, estratificado=estratificado,
                         ponderacion=ponderacion, excluir_paciente=excluir_paciente,
                         encogimiento=encogimiento, n_modelos=n_modelos, semilla=random_state)
        self.random_state = random_state

    # ``semilla`` es el nombre que usa DESCPL internamente; aquí se deriva de
    # ``random_state`` para que get_params y clone vean un solo parámetro.
    @property
    def semilla(self):
        return self.random_state

    @semilla.setter
    def semilla(self, valor):
        self.random_state = valor

    def _lista_estimadores(self):
        if self.estimadores is None:
            return estimadores_por_defecto(self.random_state)
        return self.estimadores

    def _validar(self, X, y, grupos):
        if self.estimadores is None:
            valores = np.asarray(X, dtype=float)
            if not np.isfinite(valores).all():
                raise ValueError("X contiene valores faltantes o infinitos; los modelos base por "
                                 "defecto no los imputan (usa pipelines con imputación como "
                                 "estimadores)")
        return super()._validar(X, y, grupos)
