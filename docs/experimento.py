"""
Motor del experimento combinatorio de clasificación.

Este módulo concentra la maquinaria que comparten los capítulos 6 a 9: la
construcción del preprocesador, las fábricas de modelos, las estrategias de
balanceo, los espacios de búsqueda, los cuatro optimizadores y el ejecutor
con validación cruzada anidada, agrupada por paciente y con puntos de
control en disco.

Las decisiones metodológicas se documentan en los notebooks; aquí solo está
la implementación, de modo que una corrección se aplique en un único lugar
y no en nueve capítulos.

Diseño: 7 modelos × 4 estrategias de balanceo × 4 optimizadores de
hiperparámetros = 112 corridas de clasificación. La guía del entregable
también describe una rama de regresión (7 modelos × 4 optimizadores); el
alcance de este proyecto se limitó a clasificación por decisión explícita,
así que ese módulo no la implementa.

Dependencias opcionales
-----------------------
xgboost, imbalanced-learn y optuna se importan de forma perezosa. Su
ausencia no impide usar el resto del módulo: la fábrica correspondiente
lanza un error explícito solo si se solicita ese componente.
"""

from __future__ import annotations

import importlib
import json
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (ParameterGrid, StratifiedGroupKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

#: Semilla global única, propagada a numpy, scikit-learn, XGBoost, Optuna y
#: al algoritmo genético.
SEMILLA = 42

#: Núcleos para la paralelización. -1 usa todos los disponibles.
N_JOBS = -1

# scikit-learn 1.8 marca ``penalty`` como obsoleto en favor de ``l1_ratio`` y
# avisa en cada ajuste con penalty="l1". El comportamiento no cambia (el
# ajuste sigue siendo L1), así que el aviso solo añade ruido a la salida.
warnings.filterwarnings(
    "ignore", message="Inconsistent values: penalty=.* with l1_ratio",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message="'penalty' was deprecated", category=FutureWarning,
)

class CombinacionNoAplicable(ValueError):
    """Combinación metodológicamente imposible, no un error de ejecución.

    Se distingue de ``ValueError`` para que el ejecutor registre como
    "no aplicable" solo las combinaciones que lo son por diseño (k-NN con
    class_weight), y deje propagarse cualquier error real de programación
    o de datos en lugar de ocultarlo en la tabla maestra.
    """


# ==========================================================================
# Preprocesamiento
# ==========================================================================

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


# ==========================================================================
# XGBoost con parada temprana
# ==========================================================================

class XGBClasificadorParada(ClassifierMixin, BaseEstimator):
    """XGBoost con parada temprana sobre validación interna.

    La parada temprana necesita un conjunto de validación en ``fit``, que un
    ``Pipeline`` de scikit-learn no sabe pasar a través del preprocesador.
    Este envoltorio lo resuelve dentro del propio estimador: separa una
    fracción del conjunto que recibe (ya preprocesado y, si aplica,
    remuestreado), entrena con el resto y detiene el boosting cuando la
    métrica de validación deja de mejorar durante ``rondas_parada`` rondas.

    Como recibe únicamente los datos del fold de entrenamiento, la parada
    temprana no introduce fuga hacia el fold de evaluación. La separación
    interna no está agrupada por paciente (el estimador no recibe los
    grupos), lo que puede hacer la parada algo tardía; afecta solo a cuántos
    árboles se conservan, no a la estimación del desempeño, que sigue
    viniendo del bucle externo agrupado.

    ``n_estimators`` actúa como tope; el número efectivo queda en
    ``mejor_iteracion_``.
    """

    def __init__(self, n_estimators=1000, max_depth=6, learning_rate=0.1,
                 subsample=1.0, colsample_bytree=1.0, min_child_weight=1.0,
                 scale_pos_weight=1.0, tree_method="hist",
                 fraccion_validacion=0.1, rondas_parada=30,
                 random_state=SEMILLA, n_jobs=N_JOBS):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.tree_method = tree_method
        self.fraccion_validacion = fraccion_validacion
        self.rondas_parada = rondas_parada
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _crear(self):
        xgb = _requiere("xgboost")
        return xgb.XGBClassifier(
            n_estimators=int(self.n_estimators), max_depth=int(self.max_depth),
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            scale_pos_weight=self.scale_pos_weight,
            tree_method=self.tree_method, eval_metric="aucpr",
            random_state=self.random_state, n_jobs=self.n_jobs,
            early_stopping_rounds=(self.rondas_parada
                                   if self.rondas_parada else None),
        )

    def fit(self, X, y):
        y = np.asarray(y)
        self.modelo_ = self._crear()
        if self.rondas_parada:
            X_ent, X_val, y_ent, y_val = train_test_split(
                X, y, test_size=self.fraccion_validacion,
                random_state=self.random_state, stratify=y)
            self.modelo_.fit(X_ent, y_ent, eval_set=[(X_val, y_val)],
                             verbose=False)
            self.mejor_iteracion_ = int(self.modelo_.best_iteration)
        else:
            self.modelo_.fit(X, y, verbose=False)
            self.mejor_iteracion_ = int(self.n_estimators) - 1
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.modelo_.predict(X)

    def predict_proba(self, X):
        return self.modelo_.predict_proba(X)


# ==========================================================================
# Modelos
# ==========================================================================

def _requiere(modulo, paquete=None):
    """Importa un módulo opcional con un mensaje de error accionable."""
    try:
        return importlib.import_module(modulo)
    except ImportError as exc:
        paquete = paquete or modulo
        raise ImportError(
            f"Este componente requiere '{paquete}'. Instálalo con "
            f"`pip install {paquete}` y vuelve a ejecutar la celda."
        ) from exc


def crear_modelo(nombre, semilla=SEMILLA):
    """Devuelve un estimador de clasificación sin ajustar según su nombre.

    Los siete modelos son los que exige la guía: k-NN, Naive Bayes,
    regresión logística con L1/L2, árbol de decisión, Random Forest,
    XGBoost y SVM. La guía también describe una rama de regresión (Ridge,
    Lasso, SVR...); no se implementa aquí porque el alcance del proyecto se
    limitó a clasificación.

    Decisiones de implementación (y su complejidad)
    -----------------------------------------------
    ``logistica``: solver ``liblinear`` (descenso por coordenadas / región de
    confianza), que admite L1 y L2 y opera sobre matrices dispersas. Su costo
    por iteración es O(n·p) como el de SAGA, pero converge en muchas menos
    iteraciones a esta escala: medido en el capítulo 6, es entre 10 y 15
    veces más rápido con el mismo AUC-PR.

    ``svm``: ``LinearSVC`` en lugar del SVM con kernel, cuya complejidad
    entre O(n²) y O(n³) es inviable en validación anidada con ~80,000 filas.
    Va envuelto en calibración de Platt para producir las probabilidades que
    exigen AUC-PR y el análisis de calibración.

    ``xgboost``: método ``hist`` (histogramas, O(n·p) por árbol tras una
    discretización única) en lugar del exacto (O(n·p·log n) por el
    ordenamiento), y parada temprana sobre validación interna.

    Parameters
    ----------
    nombre : {"knn", "bayes", "logistica", "arbol", "random_forest",
              "xgboost", "svm"}
    semilla : int

    Returns
    -------
    sklearn.base.BaseEstimator
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    if nombre == "knn":
        return KNeighborsClassifier(n_jobs=N_JOBS)
    if nombre == "bayes":
        return GaussianNB()
    if nombre == "logistica":
        return LogisticRegression(solver="liblinear", max_iter=1000,
                                  random_state=semilla)
    if nombre == "arbol":
        return DecisionTreeClassifier(random_state=semilla)
    if nombre == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=semilla,
                                      n_jobs=N_JOBS)
    if nombre == "xgboost":
        return XGBClasificadorParada(random_state=semilla)
    if nombre == "svm":
        return CalibratedClassifierCV(
            LinearSVC(dual="auto", max_iter=5000, random_state=semilla),
            method="sigmoid", cv=3,
        )
    raise ValueError(f"Modelo no reconocido: {nombre}")


#: Los siete modelos base de clasificación exigidos por la guía.
MODELOS = ["knn", "bayes", "logistica", "arbol", "random_forest", "xgboost",
           "svm"]

#: Las cuatro estrategias de balanceo.
BALANCEOS = ["ninguno", "smote", "adasyn", "class_weight"]

#: Los cuatro optimizadores de hiperparámetros.
OPTIMIZADORES = ["grid", "random", "optuna", "deap"]

#: Modelos que aceptan directamente ``class_weight="balanced"``.
ACEPTA_CLASS_WEIGHT = {"logistica", "arbol", "random_forest", "svm"}

#: Equivalentes de ``class_weight`` en modelos que no lo implementan:
#: GaussianNB admite fijar las probabilidades a priori y XGBoost pondera la
#: clase positiva con ``scale_pos_weight``. k-NN no tiene equivalente, ya que
#: su predicción es un voto de vecinos sin ponderación de clase.
EQUIVALENTE_CLASS_WEIGHT = {"bayes": "priors", "xgboost": "scale_pos_weight"}

#: Modelos que reciben matriz densa. GaussianNB no admite dispersas; los
#: árboles y k-NN sí, pero sus rutas dispersas son un orden de magnitud más
#: lentas (medido en el capítulo 6: ×11 árbol, ×14 Random Forest, ×4
#: inferencia de k-NN). La matriz densa ocupa ~100 MB para el conjunto de
#: entrenamiento, un costo de memoria aceptable a cambio de esa velocidad.
REQUIERE_DENSO = {"bayes", "arbol", "random_forest", "knn"}

#: Modelos que ya paralelizan internamente (``n_jobs`` propio). Para ellos
#: la búsqueda de hiperparámetros se ejecuta de forma secuencial, porque
#: paralelizar también la búsqueda crearía más hilos que núcleos
#: (sobresuscripción) sin ganancia.
PARALELO_INTERNO = {"knn", "random_forest", "xgboost"}


# ==========================================================================
# Espacios de búsqueda
# ==========================================================================
#
# Cada modelo tiene dos descripciones del mismo espacio:
#
# * Una REJILLA gruesa (listas de valores), que Grid Search recorre de forma
#   exhaustiva. Su tamaño fija el presupuesto de la corrida: 12
#   configuraciones en los modelos económicos y 8 en los costosos.
# * Un ESPACIO CONTINUO (distribuciones), del que muestrean Random Search,
#   Optuna y el algoritmo genético con ese mismo presupuesto. Es lo que
#   permite a Random Search probar valores que la rejilla nunca visita
#   (Bergstra y Bengio, 2012) y a Optuna modelar la superficie de respuesta.
#
# Formato de las distribuciones:
#   ("log", a, b)      real log-uniforme en [a, b]
#   ("float", a, b)    real uniforme en [a, b]
#   ("int", a, b)      entero uniforme en [a, b]
#   ("intlog", a, b)   entero log-uniforme en [a, b]
#   ("cat", [...])     categórica

_REJILLAS = {
    # k grande suaviza el voto; con 11 % de positivos, k pequeño deja
    # vecindarios sin casos positivos. weights="distance" da más peso a los
    # vecinos cercanos.
    "knn": {"modelo__n_neighbors": [25, 51, 101, 201],
            "modelo__weights": ["uniform", "distance"]},
    # var_smoothing estabiliza las varianzas casi nulas de las variables
    # one-hot poco frecuentes; recorre doce órdenes de magnitud.
    "bayes": {"modelo__var_smoothing": list(np.logspace(-11, 0, 12))},
    "logistica": {"modelo__C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
                  "modelo__penalty": ["l1", "l2"]},
    "arbol": {"modelo__max_depth": [4, 8, 16],
              "modelo__min_samples_leaf": [20, 100],
              "modelo__criterion": ["gini", "entropy"]},
    "random_forest": {"modelo__max_depth": [8, 16],
                      "modelo__min_samples_leaf": [5, 25],
                      "modelo__max_features": ["sqrt", 0.3]},
    "xgboost": {"modelo__max_depth": [3, 6],
                "modelo__learning_rate": [0.05, 0.2],
                "modelo__min_child_weight": [1, 10]},
    # El prefijo estimator__ atraviesa el envoltorio de calibración.
    "svm": {"modelo__estimator__C": list(np.logspace(-5, 2, 12))},
}

_DISTRIBUCIONES = {
    "knn": {"modelo__n_neighbors": ("intlog", 5, 301),
            "modelo__weights": ("cat", ["uniform", "distance"]),
            "modelo__p": ("cat", [1, 2])},
    "bayes": {"modelo__var_smoothing": ("log", 1e-12, 1.0)},
    "logistica": {"modelo__C": ("log", 1e-4, 1e2),
                  "modelo__penalty": ("cat", ["l1", "l2"])},
    "arbol": {"modelo__max_depth": ("int", 2, 30),
              "modelo__min_samples_leaf": ("intlog", 1, 500),
              "modelo__criterion": ("cat", ["gini", "entropy"]),
              "modelo__ccp_alpha": ("log", 1e-7, 1e-2)},
    "random_forest": {"modelo__n_estimators": ("int", 100, 500),
                      "modelo__max_depth": ("int", 4, 30),
                      "modelo__min_samples_leaf": ("intlog", 1, 100),
                      "modelo__max_features": ("float", 0.05, 0.6)},
    "xgboost": {"modelo__learning_rate": ("log", 0.01, 0.3),
                "modelo__max_depth": ("int", 2, 10),
                "modelo__min_child_weight": ("log", 1.0, 50.0),
                "modelo__subsample": ("float", 0.5, 1.0),
                "modelo__colsample_bytree": ("float", 0.5, 1.0)},
    "svm": {"modelo__estimator__C": ("log", 1e-6, 1e2)},
}


def rejilla_hiperparametros(nombre):
    """Rejilla gruesa que recorre Grid Search de forma exhaustiva.

    Returns
    -------
    dict
        Nombre del hiperparámetro (con prefijo ``modelo__``) a lista de
        valores.
    """
    try:
        return _REJILLAS[nombre]
    except KeyError:
        raise ValueError(f"Sin rejilla definida para: {nombre}")


def espacio_busqueda(nombre):
    """Espacio continuo del que muestrean Random, Optuna y el genético.

    Returns
    -------
    dict
        Nombre del hiperparámetro a distribución (véase el formato en el
        comentario de la sección).
    """
    try:
        return _DISTRIBUCIONES[nombre]
    except KeyError:
        raise ValueError(f"Sin espacio definido para: {nombre}")


def presupuesto_modelo(nombre):
    """Número de configuraciones por corrida: el tamaño de la rejilla.

    Los cuatro optimizadores de un mismo modelo evalúan exactamente este
    número de configuraciones, de modo que la comparación entre ellos sea a
    igual presupuesto.
    """
    return len(ParameterGrid(rejilla_hiperparametros(nombre)))


def _muestrear(distribucion, aleatorio):
    """Extrae un valor de una distribución del espacio de búsqueda."""
    tipo = distribucion[0]
    if tipo == "cat":
        opciones = distribucion[1]
        return opciones[int(aleatorio.integers(0, len(opciones)))]
    a, b = distribucion[1], distribucion[2]
    if tipo == "log":
        return float(np.exp(aleatorio.uniform(np.log(a), np.log(b))))
    if tipo == "float":
        return float(aleatorio.uniform(a, b))
    if tipo == "int":
        return int(aleatorio.integers(a, b + 1))
    if tipo == "intlog":
        return int(round(np.exp(aleatorio.uniform(np.log(a), np.log(b)))))
    raise ValueError(f"Distribución no reconocida: {distribucion}")


def _discretizar(distribucion, niveles=16):
    """Discretiza una distribución para el genoma del algoritmo genético."""
    tipo = distribucion[0]
    if tipo == "cat":
        return list(distribucion[1])
    a, b = distribucion[1], distribucion[2]
    if tipo == "log":
        return [float(v) for v in np.geomspace(a, b, niveles)]
    if tipo == "float":
        return [float(v) for v in np.linspace(a, b, niveles)]
    if tipo == "int":
        return sorted({int(round(v)) for v in np.linspace(a, b, niveles)})
    if tipo == "intlog":
        return sorted({int(round(v)) for v in np.geomspace(a, b, niveles)})
    raise ValueError(f"Distribución no reconocida: {distribucion}")


# ==========================================================================
# Pipeline y balanceo
# ==========================================================================

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
        sobremuestreo = _requiere("imblearn.over_sampling",
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


# ==========================================================================
# Métricas
# ==========================================================================

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


# ==========================================================================
# Optimizadores
# ==========================================================================

def _puntuar(pipeline, parametros, X, y, grupos, cv, metrica):
    """Puntaje medio de validación cruzada de una configuración."""
    candidato = clone(pipeline).set_params(**parametros)
    return float(np.mean(cross_val_score(
        candidato, X, y, groups=grupos, cv=cv, scoring=metrica, n_jobs=1,
        error_score="raise")))


def _puntuar_con_poda(pipeline, parametros, X, y, grupos, cv, metrica,
                      intento, optuna):
    """Evalúa fold a fold e informa a Optuna para que pueda podar.

    La fidelidad es el número de folds internos evaluados: tras cada fold se
    reporta la media acumulada y el podador decide si la configuración sigue.
    """
    from sklearn.metrics import get_scorer
    evaluador = get_scorer(metrica)
    puntajes = []
    for paso, (i_ent, i_val) in enumerate(cv.split(X, y, groups=grupos), 1):
        candidato = clone(pipeline).set_params(**parametros)
        candidato.fit(X.iloc[i_ent], y.iloc[i_ent])
        puntajes.append(evaluador(candidato, X.iloc[i_val], y.iloc[i_val]))
        intento.report(float(np.mean(puntajes)), paso)
        if intento.should_prune():
            intento.set_user_attr("folds_evaluados", paso)
            raise optuna.TrialPruned()
    intento.set_user_attr("folds_evaluados", len(puntajes))
    return float(np.mean(puntajes))


def optimizar(estrategia, pipeline, rejilla, espacio, X, y, grupos, cv,
              metrica, presupuesto, semilla=SEMILLA, paralelo_interno=False,
              multifidelidad=True):
    """Busca hiperparámetros con la estrategia indicada.

    Las cuatro estrategias evalúan el mismo número de configuraciones
    (``presupuesto``) con el mismo esquema de validación interna, de modo
    que la comparación entre ellas sea informativa.

    * ``grid``: recorre la rejilla completa.
    * ``random``: muestrea ``presupuesto`` configuraciones del espacio
      continuo.
    * ``optuna``: TPE (Tree-structured Parzen Estimator) como modelo
      sustituto, con Mejora Esperada como función de adquisición. Con
      ``multifidelidad=True`` añade Successive Halving (η = 3) sobre los folds
      internos: una configuración que tras el primer fold no está en el
      tercio superior se descarta sin evaluar los demás.
    * ``deap``: algoritmo genético sobre una discretización del espacio.

    Paralelización: si el modelo no paraleliza internamente, las
    configuraciones de Grid, Random y de cada generación del genético se
    evalúan en paralelo (configuraciones × folds). Optuna es secuencial por
    naturaleza (cada propuesta depende de las anteriores), así que en él solo
    se paralelizan los folds cuando no hay poda.

    Returns
    -------
    dict
        Mejores parámetros y puntaje, curva anytime (mejor puntaje tras cada
        evaluación), evaluaciones, diversidad genética y número de ajustes
        de pipeline consumidos.
    """
    aleatorio = np.random.default_rng(semilla)
    n_folds = cv.get_n_splits()
    evaluados, historial = [], []
    ajustes = 0

    def registrar(parametros, puntaje, folds=n_folds):
        nonlocal ajustes
        evaluados.append({**parametros, "puntaje": puntaje,
                          "folds_evaluados": folds})
        ajustes += folds
        completos = [e["puntaje"] for e in evaluados
                     if e["folds_evaluados"] == n_folds]
        historial.append(max(completos) if completos else None)

    def evaluar_lote(lista):
        """Evalúa varias configuraciones, en paralelo si conviene."""
        if not paralelo_interno and N_JOBS != 1 and len(lista) > 1:
            puntajes = Parallel(n_jobs=N_JOBS)(
                delayed(_puntuar)(pipeline, p, X, y, grupos, cv, metrica)
                for p in lista)
        else:
            puntajes = [_puntuar(pipeline, p, X, y, grupos, cv, metrica)
                        for p in lista]
        for p, s in zip(lista, puntajes):
            registrar(p, s)
        return puntajes

    diversidad = []
    if estrategia == "grid":
        candidatos = list(ParameterGrid(rejilla))
        puntajes = evaluar_lote(candidatos)
        mejor = int(np.argmax(puntajes))
        mejores, mejor_puntaje = candidatos[mejor], puntajes[mejor]

    elif estrategia == "random":
        candidatos = [{c: _muestrear(d, aleatorio) for c, d in espacio.items()}
                      for _ in range(presupuesto)]
        puntajes = evaluar_lote(candidatos)
        mejor = int(np.argmax(puntajes))
        mejores, mejor_puntaje = candidatos[mejor], puntajes[mejor]

    elif estrategia == "optuna":
        optuna = _requiere("optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def sugerir(intento):
            parametros = {}
            for clave, d in espacio.items():
                if d[0] == "cat":
                    parametros[clave] = intento.suggest_categorical(
                        clave, list(d[1]))
                elif d[0] in {"log", "float"}:
                    parametros[clave] = intento.suggest_float(
                        clave, d[1], d[2], log=d[0] == "log")
                else:
                    parametros[clave] = intento.suggest_int(
                        clave, d[1], d[2], log=d[0] == "intlog")
            return parametros

        def objetivo(intento):
            parametros = sugerir(intento)
            if multifidelidad:
                try:
                    puntaje = _puntuar_con_poda(pipeline, parametros, X, y,
                                                grupos, cv, metrica, intento,
                                                optuna)
                except optuna.TrialPruned:
                    # Configuración podada: consume los folds que alcanzó a
                    # evaluar, pero no compite por el mejor puntaje.
                    registrar(parametros, float("nan"),
                              intento.user_attrs.get("folds_evaluados", 1))
                    raise
            else:
                candidato = clone(pipeline).set_params(**parametros)
                puntaje = float(np.mean(cross_val_score(
                    candidato, X, y, groups=grupos, cv=cv, scoring=metrica,
                    n_jobs=1 if paralelo_interno else N_JOBS,
                    error_score="raise")))
            registrar(parametros, puntaje)
            return puntaje

        # TPE: modela por separado la densidad de las configuraciones buenas
        # y malas y propone donde su razón es alta, lo que equivale a
        # maximizar la Mejora Esperada. Las primeras propuestas son
        # aleatorias (n_startup_trials) para que el modelo tenga datos.
        podador = (optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=3, min_early_stopping_rate=0)
            if multifidelidad else optuna.pruners.NopPruner())
        estudio = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=semilla, n_startup_trials=max(3, presupuesto // 3)),
            pruner=podador,
        )
        estudio.optimize(objetivo, n_trials=presupuesto)
        mejores, mejor_puntaje = estudio.best_params, estudio.best_value

    elif estrategia == "deap":
        mejores, mejor_puntaje, diversidad = _algoritmo_genetico(
            espacio, evaluar_lote, presupuesto, aleatorio)

    else:
        raise ValueError(f"Estrategia no reconocida: {estrategia}")

    return {
        "mejores_parametros": mejores,
        "mejor_puntaje": mejor_puntaje,
        "historial": historial,
        "evaluaciones": evaluados,
        "diversidad": diversidad,
        "ajustes": ajustes,
    }


def _algoritmo_genetico(espacio, evaluar_lote, presupuesto, aleatorio,
                        tamano_poblacion=None, prob_cruce=0.6,
                        prob_mutacion=0.3):
    """Algoritmo genético sobre una discretización del espacio continuo.

    Se implementa con numpy en lugar de DEAP (la guía lo cita como ejemplo):
    el individuo es un vector de índices sobre la discretización de cada
    hiperparámetro (16 niveles para los continuos) y los operadores quedan
    triviales, sin la maquinaria de tipos de DEAP.

    Operadores y su justificación:

    * Selección por torneo de tamaño 3: presión selectiva moderada, sin
      requerir puntajes normalizados como la selección por ruleta.
    * Cruce uniforme (p = 0.6): apropiado cuando los genes son
      hiperparámetros independientes, sin la vecindad que supone el cruce de
      un punto.
    * Mutación por reemplazo aleatorio de un gen (p = 0.3): exploración sin
      suponer orden entre los valores.
    * Elitismo de un individuo: el mejor puntaje es monótono entre
      generaciones.
    * Población de 4 individuos (tope 8) y al menos tres generaciones dentro
      del presupuesto. Cada generación se evalúa como un lote en paralelo.

    Returns
    -------
    tuple
        Mejores parámetros, su puntaje y la diversidad genética (proporción
        de individuos distintos) por generación, indicador de convergencia
        prematura.
    """
    claves = list(espacio)
    opciones = [_discretizar(espacio[c]) for c in claves]
    limites = [len(o) for o in opciones]

    def decodificar(individuo):
        return {c: opciones[i][g] for i, (c, g) in enumerate(
            zip(claves, individuo))}

    def aleatorizar():
        return [int(aleatorio.integers(0, limite)) for limite in limites]

    if tamano_poblacion is None:
        tamano_poblacion = max(4, min(8, presupuesto // 3))
    poblacion = [aleatorizar() for _ in range(min(tamano_poblacion,
                                                  presupuesto))]
    aptitudes = evaluar_lote([decodificar(ind) for ind in poblacion])
    usadas = len(poblacion)
    diversidad = [len({tuple(i) for i in poblacion}) / len(poblacion)]
    mejor_ind, mejor_apt = None, -np.inf
    for ind, apt in zip(poblacion, aptitudes):
        if apt > mejor_apt:
            mejor_ind, mejor_apt = ind, apt

    while usadas < presupuesto:
        elite = int(np.argmax(aptitudes))
        # El élite se conserva sin reevaluarlo: no consume presupuesto.
        hijos = []
        cupo = min(len(poblacion) - 1, presupuesto - usadas)
        while len(hijos) < cupo:
            padres = []
            for _ in range(2):
                aspirantes = aleatorio.choice(len(poblacion), size=3,
                                              replace=len(poblacion) < 3)
                padres.append(poblacion[max(aspirantes,
                                            key=lambda i: aptitudes[i])])
            hijo = list(padres[0])
            if aleatorio.random() < prob_cruce:
                hijo = [padres[int(aleatorio.integers(0, 2))][j]
                        for j in range(len(claves))]
            if aleatorio.random() < prob_mutacion:
                j = int(aleatorio.integers(0, len(claves)))
                hijo[j] = int(aleatorio.integers(0, limites[j]))
            hijos.append(hijo)

        aptitudes_hijos = evaluar_lote([decodificar(h) for h in hijos])
        usadas += len(hijos)
        poblacion = [poblacion[elite]] + hijos
        aptitudes = [aptitudes[elite]] + list(aptitudes_hijos)
        diversidad.append(len({tuple(i) for i in poblacion}) / len(poblacion))
        for ind, apt in zip(hijos, aptitudes_hijos):
            if apt > mejor_apt:
                mejor_ind, mejor_apt = ind, apt

    return decodificar(mejor_ind), mejor_apt, diversidad


# ==========================================================================
# Perfilamiento
# ==========================================================================

def medir_ajuste(pipeline, X_ent, y_ent, X_eval=None, y_eval=None):
    """Mide tiempo, memoria pico y desempeño (AUC-PR) de un ajuste.

    La memoria pico se mide con ``tracemalloc``, que registra las
    asignaciones de Python y de numpy en el proceso principal; no ve la
    memoria interna de bibliotecas en C++ (XGBoost) ni la de procesos
    hijos, así que es una cota inferior en esos casos.

    Returns
    -------
    dict
        ``ajuste_s``, ``inferencia_s``, ``memoria_pico_mb`` y, si se pasa un
        conjunto de evaluación, ``auc_pr``.
    """
    tracemalloc.start()
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    ajuste = time.perf_counter() - inicio
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    resultado = {"ajuste_s": ajuste, "memoria_pico_mb": pico / 2**20}
    if X_eval is not None:
        inicio = time.perf_counter()
        salida = pipeline.predict_proba(X_eval)[:, 1]
        resultado["auc_pr"] = average_precision_score(y_eval, salida)
        resultado["inferencia_s"] = time.perf_counter() - inicio
    return resultado


# ==========================================================================
# Ejecutor con validación anidada y puntos de control
# ==========================================================================

@dataclass
class Corrida:
    """Configuración de una de las corridas del experimento.

    ``presupuesto=None`` usa el tamaño de la rejilla del modelo
    (:func:`presupuesto_modelo`), que es el valor del diseño.
    """
    modelo: str
    balanceo: str
    optimizador: str
    presupuesto: int = None
    razon_desbalance: float = None

    @property
    def identificador(self):
        return f"{self.modelo}|{self.balanceo}|{self.optimizador}"


def ejecutar_corrida(corrida, X, y, grupos, roles, folds_externos=5,
                     folds_internos=3, metrica=None, fraccion_busqueda=1.0,
                     multifidelidad=True, semilla=SEMILLA, verbose=False):
    """Valida una combinación con validación cruzada anidada y agrupada.

    El bucle externo estima el desempeño y el interno selecciona
    hiperparámetros. Ambos usan ``StratifiedGroupKFold``, de modo que los
    encuentros de un paciente nunca se reparten entre entrenamiento y
    evaluación, ni en la selección ni en la estimación.

    ``fraccion_busqueda`` < 1 hace la búsqueda sobre una submuestra de
    **pacientes** del fold externo de entrenamiento; el ajuste final usa el
    fold completo. ``multifidelidad`` activa Successive Halving en Optuna.

    Con ``verbose=True`` imprime una línea por fold externo con su duración
    y su AUC-PR. No altera ningún resultado.

    Returns
    -------
    dict
        Una fila de la tabla maestra.
    """
    metrica = metrica or METRICA_BUSQUEDA
    presupuesto = corrida.presupuesto or presupuesto_modelo(corrida.modelo)

    # Construir el pipeline una vez antes del bucle valida la combinación:
    # una combinación no aplicable se detecta aquí, antes de imprimir nada.
    construir_pipeline(corrida.modelo, corrida.balanceo, roles, semilla,
                       corrida.razon_desbalance)

    externo = StratifiedGroupKFold(n_splits=folds_externos, shuffle=True,
                                   random_state=semilla)
    interno = StratifiedGroupKFold(n_splits=folds_internos, shuffle=True,
                                   random_state=semilla)
    rejilla = rejilla_hiperparametros(corrida.modelo)
    espacio = espacio_busqueda(corrida.modelo)
    aleatorio = np.random.default_rng(semilla)
    paralelo_interno = corrida.modelo in PARALELO_INTERNO

    por_fold, elegidos, historiales, diversidades = [], [], [], []
    n_evaluaciones = n_ajustes = 0
    t_busqueda = t_ajuste = t_inferencia = 0.0

    for i, (indices_train, indices_val) in enumerate(
            externo.split(X, y, groups=grupos), 1):
        if verbose:
            print(f"  fold externo {i}/{folds_externos}...", end=" ",
                  flush=True)
        t_fold = time.perf_counter()

        X_train, X_val = X.iloc[indices_train], X.iloc[indices_val]
        y_train, y_val = y.iloc[indices_train], y.iloc[indices_val]
        grupos_train = grupos.iloc[indices_train]

        if fraccion_busqueda < 1.0:
            pacientes = grupos_train.drop_duplicates()
            elegidos_pac = aleatorio.choice(
                pacientes.to_numpy(),
                size=max(2, int(len(pacientes) * fraccion_busqueda)),
                replace=False)
            mascara = grupos_train.isin(elegidos_pac).to_numpy()
            X_busqueda, y_busqueda = X_train[mascara], y_train[mascara]
            grupos_busqueda = grupos_train[mascara]
        else:
            X_busqueda, y_busqueda = X_train, y_train
            grupos_busqueda = grupos_train

        pipeline = construir_pipeline(corrida.modelo, corrida.balanceo,
                                      roles, semilla,
                                      corrida.razon_desbalance)

        inicio = time.perf_counter()
        resultado = optimizar(
            corrida.optimizador, pipeline, rejilla, espacio, X_busqueda,
            y_busqueda, grupos_busqueda, interno, metrica, presupuesto,
            semilla=semilla + i, paralelo_interno=paralelo_interno,
            multifidelidad=multifidelidad)
        t_busqueda += time.perf_counter() - inicio

        pipeline = clone(pipeline).set_params(
            **resultado["mejores_parametros"])
        inicio = time.perf_counter()
        pipeline.fit(X_train, y_train)
        t_ajuste += time.perf_counter() - inicio

        inicio = time.perf_counter()
        salida = pipeline.predict_proba(X_val)[:, 1]
        metricas_fold = calcular_metricas(y_val, salida)
        t_inferencia += time.perf_counter() - inicio

        if verbose:
            print(f"listo en {time.perf_counter() - t_fold:.1f} s "
                  f"(auc_pr {metricas_fold['auc_pr']:.4f})", flush=True)

        por_fold.append(metricas_fold)
        elegidos.append(resultado["mejores_parametros"])
        historiales.append(resultado["historial"])
        diversidades.append(resultado["diversidad"])
        n_evaluaciones += len(resultado["evaluaciones"])
        n_ajustes += resultado["ajustes"] + 1

    fila = {
        "modelo": corrida.modelo,
        "balanceo": corrida.balanceo,
        "optimizador": corrida.optimizador,
        "presupuesto": presupuesto,
        "folds_externos": folds_externos,
        "folds_internos": folds_internos,
        "fraccion_busqueda": fraccion_busqueda,
        "multifidelidad": bool(multifidelidad and
                               corrida.optimizador == "optuna"),
        "semilla": semilla,
        "n_evaluaciones": n_evaluaciones,
        "n_ajustes": n_ajustes,
    }
    for nombre in METRICAS:
        valores = [f[nombre] for f in por_fold]
        fila[f"{nombre}_media"] = float(np.mean(valores))
        fila[f"{nombre}_sd"] = float(np.std(valores, ddof=1))
        fila[f"{nombre}_por_fold"] = json.dumps(valores)
    fila.update({
        "tiempo_busqueda_s": t_busqueda,
        "tiempo_ajuste_s": t_ajuste,
        "tiempo_inferencia_s": t_inferencia,
        "tiempo_por_evaluacion_s": t_busqueda / max(n_evaluaciones, 1),
        "hiperparametros_por_fold": json.dumps(elegidos, default=str),
        "curvas_anytime": json.dumps(historiales),
        "curvas_diversidad": json.dumps(diversidades),
    })
    return fila


def ejecutar_experimento(corridas, X, y, grupos, roles, ruta_tabla,
                         verbose=True, **kwargs):
    """Ejecuta un conjunto de corridas con puntos de control en disco.

    Cada corrida se escribe en la tabla maestra en cuanto termina, y las ya
    presentes se omiten al reanudar. Con 112 combinaciones, esto evita perder
    horas de cómputo por una interrupción.

    Parameters
    ----------
    corridas : list of Corrida
    ruta_tabla : str or pathlib.Path
        CSV de la tabla maestra. Se crea si no existe.
    kwargs
        Se pasan a :func:`ejecutar_corrida`.

    Returns
    -------
    pandas.DataFrame
        La tabla maestra completa, incluidas las corridas previas.
    """
    ruta_tabla = Path(ruta_tabla)
    if ruta_tabla.exists():
        tabla = pd.read_csv(ruta_tabla)
        hechas = set(tabla["modelo"] + "|" + tabla["balanceo"] + "|"
                     + tabla["optimizador"])
    else:
        tabla = pd.DataFrame()
        hechas = set()

    for corrida in corridas:
        if corrida.identificador in hechas:
            if verbose:
                print(f"[omitida] {corrida.identificador}")
            continue
        try:
            inicio = time.perf_counter()
            fila = ejecutar_corrida(corrida, X, y, grupos, roles,
                                    verbose=verbose, **kwargs)
            fila["tiempo_total_s"] = time.perf_counter() - inicio
            fila["estado"] = "completada"
        # Solo las combinaciones imposibles por diseño se registran como no
        # aplicables; cualquier otro error detiene la ejecución.
        except CombinacionNoAplicable as exc:
            fila = {
                "modelo": corrida.modelo, "balanceo": corrida.balanceo,
                "optimizador": corrida.optimizador,
                "estado": "no aplicable", "detalle": str(exc),
            }
        tabla = pd.concat([tabla, pd.DataFrame([fila])], ignore_index=True)
        tabla.to_csv(ruta_tabla, index=False)
        if verbose:
            estado = fila["estado"]
            detalle = (f"auc_pr {fila['auc_pr_media']:.4f} "
                       f"± {fila['auc_pr_sd']:.4f} "
                       f"en {fila['tiempo_total_s']:.1f} s"
                       if estado == "completada" else fila.get("detalle", ""))
            print(f"[{estado}] {corrida.identificador}: {detalle}")

    return tabla
