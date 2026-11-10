"""
Métricas de clasificación para clases desbalanceadas.

Forma parte de ``src/``, la versión modular del motor del experimento. Las
funciones se trasladaron sin cambios de lógica desde ``docs/experimento.py``,
que se conserva congelado para reproducir el Entregable 2; la equivalencia
entre ambos se verifica en ``tests/test_equivalencia_e2.py``.
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, precision_score, recall_score,
    roc_auc_score,
)


def calcular_metricas(y_real, probabilidades, umbral=0.5):
    """Métricas de clasificación apropiadas para clases desbalanceadas.

    El AUC-PR (precisión media) es la métrica principal junto al AUC-ROC:
    con una clase positiva del 11 %, el ROC puede parecer aceptable mientras
    la precisión es baja.
    """
    predicciones = (probabilidades >= umbral).astype(int)
    return {
        "auc_roc": roc_auc_score(y_real, probabilidades),
        "auc_pr": average_precision_score(y_real, probabilidades),
        "exactitud": accuracy_score(y_real, predicciones),
        "precision": precision_score(y_real, predicciones, zero_division=0),
        "recall": recall_score(y_real, predicciones, zero_division=0),
        "f1": f1_score(y_real, predicciones, zero_division=0),
        "exactitud_balanceada": balanced_accuracy_score(y_real, predicciones),
        "brier": brier_score_loss(y_real, probabilidades),
    }


METRICAS = ["auc_roc", "auc_pr", "exactitud", "precision", "recall", "f1",
            "exactitud_balanceada", "brier"]


#: Métrica que optimiza el bucle interno (convención de scikit-learn:
#: mayor es mejor).
METRICA_BUSQUEDA = "average_precision"


#: Métrica principal que se reporta del bucle externo.
METRICA_PRINCIPAL = "auc_pr"
