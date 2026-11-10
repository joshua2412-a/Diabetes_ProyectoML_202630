"""
Estrategias de balanceo y ensamblaje del pipeline completo
(preprocesamiento, remuestreo y modelo).

Forma parte de ``src/``, la versión modular del motor del experimento. Las
funciones se trasladaron sin cambios de lógica desde ``docs/experimento.py``,
que se conserva congelado para reproducir el Entregable 2; la equivalencia
entre ambos se verifica en ``tests/test_equivalencia_e2.py``.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from .configuracion import SEMILLA, CombinacionNoAplicable, requiere
from .modelos import ACEPTA_CLASS_WEIGHT, REQUIERE_DENSO, crear_modelo
from .preprocesamiento import construir_preprocesador

#: Las cuatro estrategias de balanceo.
BALANCEOS = ["ninguno", "smote", "adasyn", "class_weight"]


def construir_pipeline(modelo, balanceo, roles, semilla=SEMILLA,
                       razon_desbalance=None, salida_densa=None):
    """Ensambla preprocesador, remuestreo y modelo en un único pipeline.

    El remuestreo se coloca **dentro** del pipeline y después del
    preprocesador, lo que garantiza que las muestras sintéticas se generen
    solo con los datos del fold de entrenamiento.

    Parameters
    ----------
    modelo : str
    balanceo : {"ninguno", "smote", "adasyn", "class_weight"}
    roles : dict
    semilla : int
    razon_desbalance : float, optional
        Necesaria para el equivalente de ``class_weight`` en XGBoost.
    salida_densa : bool, optional
        Fuerza la representación de la matriz. Por defecto se decide según
        :data:`REQUIERE_DENSO`. Solo se usa para comparar representaciones.

    Raises
    ------
    CombinacionNoAplicable
        Si la combinación es imposible por diseño.
    """
    estimador = crear_modelo(modelo, semilla)
    if salida_densa is None:
        salida_densa = modelo in REQUIERE_DENSO
    pasos = [("preprocesamiento", construir_preprocesador(
        roles, salida_densa=salida_densa))]

    if balanceo == "class_weight":
        if modelo in ACEPTA_CLASS_WEIGHT:
            if modelo == "svm":
                estimador.estimator.set_params(class_weight="balanced")
            else:
                estimador.set_params(class_weight="balanced")
        elif modelo == "bayes":
            # Probabilidades a priori uniformes: eliminan la ventaja de la
            # clase mayoritaria en el producto posterior.
            estimador.set_params(priors=[0.5, 0.5])
        elif modelo == "xgboost":
            if razon_desbalance is None:
                raise ValueError(
                    "XGBoost necesita razon_desbalance para emular "
                    "class_weight mediante scale_pos_weight.")
            estimador.set_params(scale_pos_weight=razon_desbalance)
        else:
            raise CombinacionNoAplicable(
                f"'{modelo}' no tiene equivalente de class_weight: su "
                "predicción es un voto de vecinos sin ponderación de clase.")

    if balanceo in {"smote", "adasyn"}:
        sobremuestreo = requiere("imblearn.over_sampling",
                                  "imbalanced-learn")
        clase = getattr(sobremuestreo, "SMOTE" if balanceo == "smote"
                        else "ADASYN")
        pasos.append(("balanceo", clase(random_state=semilla)))
        from imblearn.pipeline import Pipeline as PipelineIMB
        pasos.append(("modelo", estimador))
        return PipelineIMB(pasos)

    if balanceo not in {"ninguno", "class_weight"}:
        raise ValueError(f"Balanceo no reconocido: {balanceo}")

    pasos.append(("modelo", estimador))
    return Pipeline(pasos)
