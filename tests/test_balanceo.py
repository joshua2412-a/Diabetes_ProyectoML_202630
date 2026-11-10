"""Pruebas de las estrategias de balanceo y del pipeline (src/balanceo.py)."""

import pytest

from src.balanceo import construir_pipeline
from src.configuracion import CombinacionNoAplicable


@pytest.mark.parametrize("balanceo", ["smote", "adasyn"])
def test_remuestreo_despues_del_preprocesador_y_en_imblearn(roles, balanceo):
    pipeline = construir_pipeline("logistica", balanceo, roles)
    assert [n for n, _ in pipeline.steps] == ["preprocesamiento", "balanceo",
                                              "modelo"]
    # El Pipeline de imbalanced-learn aplica el remuestreo solo en fit.
    assert type(pipeline).__module__.startswith("imblearn")


def test_smote_solo_actua_al_entrenar(datos, roles):
    """El modelo se entrena con más filas de las recibidas, pero predice
    exactamente una probabilidad por fila: SMOTE no actúa al predecir."""
    X, y, _ = datos
    pipeline = construir_pipeline("arbol", "smote", roles).fit(X, y)
    filas_entrenamiento = pipeline.named_steps["modelo"].tree_.n_node_samples[0]
    assert filas_entrenamiento > len(X)
    assert pipeline.predict_proba(X).shape == (len(X), 2)


def test_sin_balanceo_no_cambia_las_filas(datos, roles):
    X, y, _ = datos
    pipeline = construir_pipeline("arbol", "ninguno", roles).fit(X, y)
    assert pipeline.named_steps["modelo"].tree_.n_node_samples[0] == len(X)


def test_equivalentes_de_class_weight(roles):
    modelo = lambda p: p.named_steps["modelo"]
    assert modelo(construir_pipeline("logistica", "class_weight", roles)) \
        .class_weight == "balanced"
    assert modelo(construir_pipeline("svm", "class_weight", roles)) \
        .estimator.class_weight == "balanced"
    assert list(modelo(construir_pipeline("bayes", "class_weight", roles))
                .priors) == [0.5, 0.5]
    xgb = modelo(construir_pipeline("xgboost", "class_weight", roles,
                                    razon_desbalance=4.0))
    assert xgb.scale_pos_weight == 4.0


def test_xgboost_class_weight_exige_la_razon(roles):
    with pytest.raises(ValueError) as error:
        construir_pipeline("xgboost", "class_weight", roles)
    # Es un error de uso, no una combinación imposible por diseño.
    assert not isinstance(error.value, CombinacionNoAplicable)


def test_knn_con_class_weight_no_es_aplicable(roles):
    with pytest.raises(CombinacionNoAplicable):
        construir_pipeline("knn", "class_weight", roles)


def test_balanceo_desconocido(roles):
    with pytest.raises(ValueError):
        construir_pipeline("logistica", "inventado", roles)
