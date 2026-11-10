"""
Benchmarks recientes del estado del arte (guía del Entregable 3, sección 6.2).

La guía exige comparar el modelo original, además de con los siete modelos
vistos en clase, con al menos dos métodos recientes del estado del arte para
la misma tarea. Para clasificación tabular se eligen los dos boosting de
árboles que la literatura comparativa señala como referencia en datos
tabulares de tamaño medio (Grinsztajn, Oyallon y Varoquaux, 2022; Shwartz-Ziv
y Armon, 2022), ambos no neuronales:

* **LightGBM** (Ke et al., 2017): crecimiento de árboles por hoja e
  histogramas.
* **CatBoost** (Prokhorenkova et al., 2018): boosting ordenado, que reduce
  el sesgo de predicción del gradient boosting clásico.

Reciben exactamente el mismo protocolo que los modelos base en el estudio de
convergencia del capítulo 12: mismo preprocesador, mismas particiones
externas e internas, Random Search con 60 evaluaciones sobre un cuarto de los
pacientes (como XGBoost y Random Forest) y reajuste con todo el fold. El
desbalance se trata como un hiperparámetro más (pesos por clase en LightGBM;
peso de la clase positiva entre 1 y 10 en CatBoost), de modo que la búsqueda
elige lo que mejor funciona. Los espacios parten de los
valores recomendados por sus autores y por la documentación.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from src import configuracion as cfg
from src.preprocesamiento import construir_preprocesador

MODELOS_SOTA = ("lightgbm", "catboost")

#: Fracción de pacientes con la que se busca, igual que XGBoost y Random
#: Forest en el capítulo 12.
FRACCION_BUSQUEDA = 0.25

#: Formato de las distribuciones: el de ``src.espacios``.
ESPACIOS = {
    "lightgbm": {
        "modelo__n_estimators": ("int", 100, 1000),
        "modelo__learning_rate": ("log", 0.01, 0.3),
        "modelo__num_leaves": ("intlog", 8, 256),
        "modelo__min_child_samples": ("intlog", 5, 300),
        "modelo__subsample": ("float", 0.5, 1.0),
        "modelo__colsample_bytree": ("float", 0.5, 1.0),
        "modelo__reg_lambda": ("log", 1e-3, 10.0),
        "modelo__class_weight": ("cat", [None, "balanced"]),
    },
    "catboost": {
        "modelo__iterations": ("int", 200, 800),
        "modelo__learning_rate": ("log", 0.01, 0.3),
        "modelo__depth": ("int", 4, 8),
        "modelo__l2_leaf_reg": ("log", 1.0, 30.0),
        # CatBoost no acepta auto_class_weights=None vía set_params; el peso
        # de la clase positiva cubre lo mismo: 1 es sin pesos y la razón de
        # desbalance de estos datos (~7,8) equivale a "Balanced".
        "modelo__scale_pos_weight": ("log", 1.0, 10.0),
    },
}


def crear_modelo_sota(nombre, semilla=cfg.SEMILLA):
    """Estimador con los valores fijos que no se buscan."""
    if nombre == "lightgbm":
        lightgbm = cfg.requiere("lightgbm")
        return lightgbm.LGBMClassifier(subsample_freq=1, random_state=semilla,
                                       n_jobs=cfg.N_JOBS, verbose=-1)
    if nombre == "catboost":
        catboost = cfg.requiere("catboost")
        return catboost.CatBoostClassifier(random_seed=semilla, thread_count=cfg.N_JOBS,
                                           verbose=0, allow_writing_files=False)
    raise ValueError(f"Benchmark desconocido: {nombre}; usa {MODELOS_SOTA}")


def construir_pipeline_sota(nombre, roles, semilla=cfg.SEMILLA):
    """Mismo preprocesador que los modelos base, con salida densa."""
    return Pipeline([("preprocesamiento", construir_preprocesador(roles, salida_densa=True)),
                     ("modelo", crear_modelo_sota(nombre, semilla))])
