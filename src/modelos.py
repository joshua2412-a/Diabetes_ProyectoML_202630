"""
Los siete modelos base, el envoltorio de XGBoost con parada temprana y los
catálogos que describen cómo trata el diseño a cada modelo.

Forma parte de ``src/``, la versión modular del motor del experimento. Las
funciones se trasladaron sin cambios de lógica desde ``docs/experimento.py``,
que se conserva congelado para reproducir el Entregable 2; la equivalencia
entre ambos se verifica en ``tests/test_equivalencia_e2.py``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from . import configuracion as cfg
from .configuracion import SEMILLA, requiere


# El valor por defecto de ``n_jobs`` (-1, todos los núcleos) es el que tenía
# ``N_JOBS`` al importar el módulo original. ``crear_modelo`` siempre pasa el
# valor vigente de forma explícita, así que el diseño no depende de él.
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

    Hay una consecuencia adicional que conviene declarar de forma explícita.
    Cuando la combinación incluye SMOTE o ADASYN, los datos que llegan a
    ``fit`` ya están remuestreados, de modo que el conjunto de validación
    interno contiene observaciones sintéticas interpoladas a partir de la
    porción de entrenamiento. Al ser interpolaciones son más fáciles de
    ajustar que observaciones reales, así que la curva de validación se ve
    mejor por más tiempo y la parada dispara más tarde de lo que haría sobre
    datos reales: el modelo queda algo sobredimensionado en número de
    árboles. Afecta a ocho de las ciento ocho corridas del diseño (XGBoost
    con SMOTE y con ADASYN, por los cuatro optimizadores) y no contamina la
    estimación del desempeño, porque el recorte se hace enteramente dentro
    del fold de entrenamiento. La cota del efecto es acotable con los datos
    del propio experimento: la mejor corrida de XGBoost afectada alcanza un
    AUC-PR de 0.2267 y la mejor no afectada 0.2243, de modo que la
    corrección máxima posible es de 0.0024, por debajo del rango de 0.0041
    que produce cambiar la semilla aleatoria.

    Corregirlo exigiría separar la validación interna antes del remuestreo,
    lo que implica reejecutar esas ocho corridas y, con ellas, los capítulos
    9 a 11. Se documenta en lugar de corregirse porque la cota anterior no
    altera ninguna conclusión del proyecto.

    ``n_estimators`` actúa como tope; el número efectivo queda en
    ``mejor_iteracion_``.
    """

    def __init__(self, n_estimators=1000, max_depth=6, learning_rate=0.1,
                 subsample=1.0, colsample_bytree=1.0, min_child_weight=1.0,
                 scale_pos_weight=1.0, tree_method="hist",
                 fraccion_validacion=0.1, rondas_parada=30,
                 random_state=SEMILLA, n_jobs=-1):
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
        xgb = requiere("xgboost")
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
    iteraciones a esta escala: medido en el capítulo 6, el ajuste del
    pipeline completo es 6.7 veces más rápido con el mismo AUC-PR (la mejora
    del solver aislado es mayor, porque el preprocesamiento es común a ambos).

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
        return KNeighborsClassifier(n_jobs=cfg.N_JOBS)
    if nombre == "bayes":
        return GaussianNB()
    if nombre == "logistica":
        return LogisticRegression(solver="liblinear", max_iter=1000,
                                  random_state=semilla)
    if nombre == "arbol":
        return DecisionTreeClassifier(random_state=semilla)
    if nombre == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=semilla,
                                      n_jobs=cfg.N_JOBS)
    if nombre == "xgboost":
        # n_jobs se pasa explícito: el valor por defecto del parámetro en
        # XGBClasificadorParada se fija al importar el módulo y no reflejaría
        # un cambio posterior de N_JOBS (por ejemplo, ``cfg.N_JOBS = 4``).
        return XGBClasificadorParada(random_state=semilla, n_jobs=cfg.N_JOBS)
    if nombre == "svm":
        return CalibratedClassifierCV(
            LinearSVC(dual="auto", max_iter=5000, random_state=semilla),
            method="sigmoid", cv=3,
        )
    raise ValueError(f"Modelo no reconocido: {nombre}")


#: Los siete modelos base de clasificación exigidos por la guía.
MODELOS = ["knn", "bayes", "logistica", "arbol", "random_forest", "xgboost",
           "svm"]


#: Modelos que aceptan directamente ``class_weight="balanced"``.
ACEPTA_CLASS_WEIGHT = {"logistica", "arbol", "random_forest", "svm"}


#: Equivalentes de ``class_weight`` en modelos que no lo implementan:
#: GaussianNB admite fijar las probabilidades a priori y XGBoost pondera la
#: clase positiva con ``scale_pos_weight``. k-NN no tiene equivalente, ya que
#: su predicción es un voto de vecinos sin ponderación de clase.
EQUIVALENTE_CLASS_WEIGHT = {"bayes": "priors", "xgboost": "scale_pos_weight"}


#: Modelos que reciben matriz densa. GaussianNB no admite dispersas; los
#: árboles y k-NN sí, pero sus rutas dispersas son mucho más lentas (medido
#: en el capítulo 6 sobre el pipeline completo: ×2.4 el árbol, ×3.6 Random
#: Forest y ×6.3 la inferencia de k-NN, con el mismo AUC-PR). La matriz densa
#: cuesta en torno a un 10 % más de memoria pico, a cambio de esa velocidad.
REQUIERE_DENSO = {"bayes", "arbol", "random_forest", "knn"}


#: Modelos que ya paralelizan internamente (``n_jobs`` propio). Para ellos
#: la búsqueda de hiperparámetros se ejecuta de forma secuencial, porque
#: paralelizar también la búsqueda crearía más hilos que núcleos
#: (sobresuscripción) sin ganancia.
PARALELO_INTERNO = {"knn", "random_forest", "xgboost"}
