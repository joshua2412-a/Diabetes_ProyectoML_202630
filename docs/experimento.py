"""
Motor del experimento combinatorio de clasificación.

Este módulo concentra la maquinaria que comparten los capítulos 6 a 9: la
construcción del preprocesador, las fábricas de modelos, estrategias de
balanceo y optimizadores, y el ejecutor con validación cruzada anidada y
puntos de control.

Las decisiones metodológicas se documentan en los notebooks; aquí solo está
la implementación, de modo que una corrección se aplique en un único lugar
y no en nueve capítulos.

Dependencias opcionales
-----------------------
xgboost, imbalanced-learn, optuna y deap se importan de forma perezosa. Su
ausencia no impide usar el resto del módulo: la fábrica correspondiente
lanza un error explícito solo si se solicita ese componente.
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

SEMILLA = 42


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
        Si es ``True``, la codificación one-hot devuelve una matriz densa.
        Necesario para los estimadores que no admiten matrices dispersas,
        como ``GaussianNB``.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocesador sin ajustar.
    """
    numericas = roles["numericas"] + roles["ordinales"]
    binarias = roles["binarias"]
    categoricas = roles["categoricas"]

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

    return ColumnTransformer([
        ("num", bloque_numerico, numericas),
        ("bin", "passthrough", binarias),
        ("cat", bloque_categorico, categoricas),
    ], verbose_feature_names_out=False)


def nombres_variables(preprocesador):
    """Nombres de las columnas que produce un preprocesador ya ajustado."""
    return list(preprocesador.get_feature_names_out())


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
    """Devuelve un estimador sin ajustar según su nombre corto.

    Los siete modelos son los exigidos por la guía del entregable: k-NN,
    Naive Bayes, regresión logística con regularización L1/L2, árbol de
    decisión, Random Forest, XGBoost y SVM. La regularización L1 y L2 de la
    logística se trata como un hiperparámetro y no como dos modelos
    distintos, de modo que la búsqueda decida entre penalización dispersa y
    densa con el mismo presupuesto.

    Notas de implementación
    -----------------------
    ``logistica`` usa el solver SAGA, que tiene complejidad lineal en n y p
    por iteración, admite ambas penalizaciones y opera sobre matrices
    dispersas: la combinación adecuada para 79,473 filas y 129 columnas
    codificadas.

    ``svm`` usa ``LinearSVC`` envuelto en calibración en lugar de ``SVC`` con
    kernel: el SVM con kernel tiene complejidad entre O(n^2) y O(n^3) en el
    número de observaciones, inviable dentro de una validación anidada a esta
    escala. La calibración de Platt aporta además las probabilidades que
    exigen el AUC-PR y el análisis de calibración del capítulo 9.

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
        return KNeighborsClassifier(n_jobs=-1)
    if nombre == "bayes":
        return GaussianNB()
    if nombre == "logistica":
        return LogisticRegression(solver="saga", max_iter=3000,
                                  random_state=semilla)
    if nombre == "arbol":
        return DecisionTreeClassifier(random_state=semilla)
    if nombre == "random_forest":
        return RandomForestClassifier(random_state=semilla, n_jobs=-1)
    if nombre == "xgboost":
        xgb = _requiere("xgboost")
        return xgb.XGBClassifier(
            random_state=semilla, tree_method="hist", eval_metric="aucpr",
            n_jobs=-1,
        )
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

#: Modelos sensibles a la escala de las variables.
SENSIBLE_A_ESCALA = {"logistica", "knn", "svm", "bayes"}

#: Estimadores que no admiten matrices dispersas y exigen codificación densa.
REQUIERE_DENSO = {"bayes"}


def espacio_hiperparametros(nombre):
    """Rejilla de hiperparámetros de un modelo, con rangos justificados.

    Los rangos se fijan sobre escalas logarítmicas en los parámetros de
    regularización, donde el óptimo puede estar en cualquier orden de
    magnitud, y sobre valores discretos interpretables en los parámetros
    estructurales de los árboles. Las claves usan el prefijo ``modelo__``
    para encajar en el pipeline.

    Returns
    -------
    dict
        Nombre del hiperparámetro a lista de valores.
    """
    espacios = {
        # k impar evita empates. El rango va de memorización local (5) a
        # promediado amplio (101), apropiado con 79,473 filas. La distancia de
        # Manhattan suele degradarse menos que la euclídea en dimensión alta.
        "knn": {
            "modelo__n_neighbors": [5, 15, 31, 51, 101],
            "modelo__weights": ["uniform", "distance"],
            "modelo__p": [1, 2],
        },
        # var_smoothing añade una fracción de la varianza máxima a todas las
        # varianzas, lo que estabiliza las variables binarias casi constantes
        # que produce la codificación one-hot.
        "bayes": {"modelo__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
        # C recorre cinco órdenes de magnitud; la penalización L1 frente a L2
        # decide entre selección dispersa de variables y encogimiento denso.
        "logistica": {
            "modelo__C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "modelo__penalty": ["l1", "l2"],
        },
        # ccp_alpha implementa la poda de complejidad de costo, que es la vía
        # principal de regularización del árbol; min_samples_leaf evita hojas
        # con un solo paciente, frecuentes con 11 % de clase positiva.
        "arbol": {
            "modelo__max_depth": [3, 6, 12, None],
            "modelo__min_samples_leaf": [10, 50, 200],
            "modelo__criterion": ["gini", "entropy"],
            "modelo__ccp_alpha": [0.0, 1e-4, 1e-3],
        },
        "random_forest": {
            "modelo__n_estimators": [200, 400],
            "modelo__max_depth": [6, 12, None],
            "modelo__min_samples_leaf": [1, 10, 50],
            "modelo__max_features": ["sqrt", 0.5],
        },
        "xgboost": {
            "modelo__n_estimators": [200, 400],
            "modelo__max_depth": [3, 6, 9],
            "modelo__learning_rate": [0.03, 0.1, 0.3],
            "modelo__subsample": [0.7, 1.0],
            "modelo__colsample_bytree": [0.7, 1.0],
            "modelo__min_child_weight": [1, 10],
        },
        # El prefijo estimator__ atraviesa el envoltorio de calibración.
        "svm": {"modelo__estimator__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    }
    if nombre not in espacios:
        raise ValueError(f"Sin espacio definido para: {nombre}")
    return espacios[nombre]


# ==========================================================================
# Balanceo
# ==========================================================================

def construir_pipeline(modelo, balanceo, roles, semilla=SEMILLA,
                       razon_desbalance=None):
    """Ensambla preprocesador, remuestreo y modelo en un único pipeline.

    El remuestreo se coloca **dentro** del pipeline y después del
    preprocesador, lo que garantiza dos cosas: que las muestras sintéticas se
    generen solo con los datos del fold de entrenamiento, y que se generen
    sobre la matriz ya codificada, que es donde las distancias entre vecinos
    tienen sentido.

    Parameters
    ----------
    modelo : str
        Nombre del modelo, según :func:`crear_modelo`.
    balanceo : {"ninguno", "smote", "adasyn", "class_weight"}
    roles : dict
    semilla : int
    razon_desbalance : float, optional
        Razón entre la clase mayoritaria y la minoritaria, necesaria para el
        equivalente de ``class_weight`` en XGBoost (``scale_pos_weight``). Si
        se omite y se solicita esa combinación, se lanza un error en lugar de
        asumir un valor arbitrario.

    Returns
    -------
    sklearn.pipeline.Pipeline or imblearn.pipeline.Pipeline

    Raises
    ------
    ValueError
        Si la combinación no es aplicable, como ``class_weight`` en un modelo
        que no acepta ese parámetro.
    """
    estimador = crear_modelo(modelo, semilla)
    pasos = [("preprocesamiento", construir_preprocesador(
        roles, salida_densa=modelo in REQUIERE_DENSO))]

    if balanceo == "class_weight":
        if modelo in ACEPTA_CLASS_WEIGHT:
            if modelo == "svm":
                estimador.estimator.set_params(class_weight="balanced")
            else:
                estimador.set_params(class_weight="balanced")
        elif modelo == "bayes":
            # Equivalente en Naive Bayes: probabilidades a priori uniformes,
            # que eliminan la ventaja de la clase mayoritaria en el producto
            # posterior.
            estimador.set_params(priors=[0.5, 0.5])
        elif modelo == "xgboost":
            if razon_desbalance is None:
                raise ValueError(
                    "XGBoost necesita razon_desbalance para emular "
                    "class_weight mediante scale_pos_weight."
                )
            estimador.set_params(scale_pos_weight=razon_desbalance)
        else:
            raise ValueError(
                f"'{modelo}' no tiene equivalente de class_weight: su "
                "predicción es un voto de vecinos sin ponderación de clase. "
                "La combinación se registra como no aplicable."
            )

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
    la precisión es baja, porque el eje de falsos positivos se normaliza por
    una clase mayoritaria muy grande.

    Returns
    -------
    dict
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


# ==========================================================================
# Optimizadores
# ==========================================================================

def _muestrear_rejilla(espacio, n, aleatorio):
    """Extrae n combinaciones distintas de una rejilla, sin reemplazo."""
    from sklearn.model_selection import ParameterGrid
    combinaciones = list(ParameterGrid(espacio))
    if n >= len(combinaciones):
        return combinaciones
    indices = aleatorio.choice(len(combinaciones), size=n, replace=False)
    return [combinaciones[i] for i in indices]


def optimizar(estrategia, pipeline, espacio, X, y, grupos, cv, metrica,
              presupuesto, semilla=SEMILLA):
    """Busca hiperparámetros con la estrategia indicada.

    Las cuatro estrategias comparten presupuesto y esquema de validación
    interna, de modo que la comparación entre ellas sea informativa: si una
    gana, es por su forma de explorar el espacio y no porque haya evaluado
    más candidatos.

    Parameters
    ----------
    estrategia : {"grid", "random", "optuna", "deap"}
    pipeline : sklearn pipeline sin ajustar
    espacio : dict
    X, y : datos del fold externo de entrenamiento
    grupos : array-like
        Identificador de paciente, para que la validación interna agrupe.
    cv : validador con soporte de grupos
    metrica : str
        Nombre de la métrica de scikit-learn a maximizar.
    presupuesto : int
        Número de configuraciones a evaluar.
    semilla : int

    Returns
    -------
    dict
        ``mejores_parametros``, ``mejor_puntaje``, ``historial`` (curva
        anytime: mejor puntaje acumulado tras cada evaluación) y
        ``evaluaciones``.
    """
    from sklearn.model_selection import cross_val_score

    aleatorio = np.random.default_rng(semilla)
    historial, evaluados, diversidad = [], [], []

    def evaluar(parametros):
        """Puntaje medio de una configuración en la validación interna."""
        candidato = pipeline.set_params(**parametros)
        puntajes = cross_val_score(candidato, X, y, groups=grupos, cv=cv,
                                   scoring=metrica, n_jobs=1)
        puntaje = float(np.mean(puntajes))
        evaluados.append({**parametros, "puntaje": puntaje})
        historial.append(max(h for h in [puntaje] + [
            e["puntaje"] for e in evaluados]))
        return puntaje

    if estrategia == "grid":
        candidatos = _muestrear_rejilla(espacio, presupuesto, aleatorio)
        puntajes = [evaluar(c) for c in candidatos]
        mejor = int(np.argmax(puntajes))
        mejores, mejor_puntaje = candidatos[mejor], puntajes[mejor]

    elif estrategia == "random":
        candidatos = _muestrear_rejilla(espacio, presupuesto, aleatorio)
        aleatorio.shuffle(candidatos)
        puntajes = [evaluar(c) for c in candidatos]
        mejor = int(np.argmax(puntajes))
        mejores, mejor_puntaje = candidatos[mejor], puntajes[mejor]

    elif estrategia == "optuna":
        optuna = _requiere("optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objetivo(intento):
            parametros = {
                clave: intento.suggest_categorical(clave, list(valores))
                for clave, valores in espacio.items()
            }
            return evaluar(parametros)

        # TPE como modelo sustituto: estima densidades de las configuraciones
        # buenas y malas por separado y muestrea donde su razón es alta, lo
        # que se adapta bien a espacios mixtos y discretos como este.
        estudio = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=semilla),
        )
        estudio.optimize(objetivo, n_trials=presupuesto)
        mejores, mejor_puntaje = estudio.best_params, estudio.best_value

    elif estrategia == "deap":
        mejores, mejor_puntaje, diversidad = _algoritmo_genetico(
            espacio, evaluar, presupuesto, aleatorio)

    else:
        raise ValueError(f"Estrategia no reconocida: {estrategia}")

    return {
        "mejores_parametros": mejores,
        "mejor_puntaje": mejor_puntaje,
        "historial": historial,
        "evaluaciones": evaluados,
        "diversidad": diversidad if estrategia == "deap" else [],
    }


def _algoritmo_genetico(espacio, evaluar, presupuesto, aleatorio,
                        tamano_poblacion=None, prob_cruce=0.6,
                        prob_mutacion=0.3):
    """Algoritmo genético sobre un espacio de hiperparámetros discreto.

    Se implementa con numpy en lugar de DEAP porque el espacio es un producto
    cartesiano de listas: el individuo es un vector de índices y los operadores
    quedan triviales, sin la maquinaria de tipos de DEAP. El comportamiento es
    el de un algoritmo genético estándar con elitismo.

    Operadores y su justificación:

    * Selección por torneo de tamaño 3: mantiene presión selectiva moderada y
      no requiere puntajes normalizados, a diferencia de la selección
      proporcional.
    * Cruce uniforme: apropiado cuando los genes (hiperparámetros) son
      independientes entre sí, sin la estructura de vecindad que asumiría un
      cruce de un punto.
    * Mutación por reemplazo aleatorio de un gen: exploración local sin
      suponer orden entre los valores de cada hiperparámetro.
    * Elitismo de un individuo: garantiza que el mejor puntaje sea monótono
      entre generaciones.

    Returns
    -------
    tuple
        Mejores parámetros, su puntaje y la curva de diversidad genética
        (proporción de individuos distintos en cada generación), que permite
        detectar convergencia prematura.
    """
    claves = list(espacio)
    opciones = [list(espacio[c]) for c in claves]
    limites = [len(o) for o in opciones]

    def decodificar(individuo):
        return {c: opciones[i][g] for i, (c, g) in enumerate(
            zip(claves, individuo))}

    def aleatorizar():
        return [int(aleatorio.integers(0, limite)) for limite in limites]

    # El tamaño de población se adapta al presupuesto para que haya al menos
    # tres generaciones: con una sola generación el algoritmo degenera en una
    # búsqueda aleatoria y la curva de diversidad no informa de nada.
    if tamano_poblacion is None:
        tamano_poblacion = max(4, min(8, presupuesto // 3))
    poblacion = [aleatorizar() for _ in range(min(tamano_poblacion,
                                                  presupuesto))]
    aptitudes = [evaluar(decodificar(ind)) for ind in poblacion]
    usadas = len(poblacion)
    diversidad = [len({tuple(i) for i in poblacion}) / len(poblacion)]

    while usadas < presupuesto:
        elite = int(np.argmax(aptitudes))
        nueva = [poblacion[elite]]

        while len(nueva) < len(poblacion) and usadas + len(nueva) <= presupuesto:
            # Selección por torneo
            padres = []
            for _ in range(2):
                aspirantes = aleatorio.choice(len(poblacion), size=3,
                                              replace=False)
                padres.append(poblacion[max(aspirantes,
                                            key=lambda i: aptitudes[i])])
            hijo = list(padres[0])
            if aleatorio.random() < prob_cruce:      # cruce uniforme
                hijo = [padres[aleatorio.integers(0, 2)][j]
                        for j in range(len(claves))]
            if aleatorio.random() < prob_mutacion:   # mutación puntual
                j = int(aleatorio.integers(0, len(claves)))
                hijo[j] = int(aleatorio.integers(0, limites[j]))
            nueva.append(hijo)

        poblacion = nueva
        aptitudes = [evaluar(decodificar(ind)) for ind in poblacion]
        usadas += len(poblacion)
        diversidad.append(len({tuple(i) for i in poblacion}) / len(poblacion))

    mejor = int(np.argmax(aptitudes))
    return decodificar(poblacion[mejor]), aptitudes[mejor], diversidad


# ==========================================================================
# Ejecutor con validación anidada y puntos de control
# ==========================================================================

@dataclass
class Corrida:
    """Configuración de una de las corridas del experimento."""
    modelo: str
    balanceo: str
    optimizador: str
    presupuesto: int = 12
    razon_desbalance: float = None

    @property
    def identificador(self):
        return f"{self.modelo}|{self.balanceo}|{self.optimizador}"


def ejecutar_corrida(corrida, X, y, grupos, roles, folds_externos=5,
                     folds_internos=3, metrica="average_precision",
                     fraccion_busqueda=1.0, semilla=SEMILLA):
    """Valida una combinación con validación cruzada anidada y agrupada.

    El bucle externo estima el desempeño y el interno selecciona
    hiperparámetros. Ambos usan ``StratifiedGroupKFold``, de modo que los
    encuentros de un paciente nunca se reparten entre entrenamiento y
    evaluación, ni en la selección ni en la estimación.

    El argumento ``fraccion_busqueda`` implementa la estrategia de
    multi-fidelidad: la búsqueda de hiperparámetros se ejecuta sobre una
    submuestra de **pacientes** del fold externo de entrenamiento, mientras
    que el ajuste final de ese fold usa todos sus datos. Submuestrear por
    paciente y no por fila mantiene la integridad de los grupos también en la
    búsqueda. El supuesto implícito es que el orden relativo de las
    configuraciones se conserva al reducir el tamaño de la muestra.

    Returns
    -------
    dict
        Una fila de la tabla maestra: métricas como media y desviación
        estándar entre folds externos, tiempos separados, los hiperparámetros
        elegidos en cada fold y las curvas anytime de la búsqueda.
    """
    externo = StratifiedGroupKFold(n_splits=folds_externos, shuffle=True,
                                   random_state=semilla)
    interno = StratifiedGroupKFold(n_splits=folds_internos, shuffle=True,
                                   random_state=semilla)

    espacio = espacio_hiperparametros(corrida.modelo)
    aleatorio = np.random.default_rng(semilla)
    por_fold, elegidos, historiales, diversidades = [], [], [], []
    t_busqueda = t_ajuste = t_inferencia = 0.0

    for indices_train, indices_val in externo.split(X, y, groups=grupos):
        X_train, X_val = X.iloc[indices_train], X.iloc[indices_val]
        y_train, y_val = y.iloc[indices_train], y.iloc[indices_val]
        grupos_train = grupos.iloc[indices_train]

        # Multi-fidelidad: submuestra de pacientes para la búsqueda interna.
        if fraccion_busqueda < 1.0:
            pacientes = grupos_train.drop_duplicates()
            elegidos_pac = aleatorio.choice(
                pacientes.to_numpy(),
                size=max(2, int(len(pacientes) * fraccion_busqueda)),
                replace=False)
            mascara = grupos_train.isin(elegidos_pac).to_numpy()
            X_busqueda = X_train[mascara]
            y_busqueda = y_train[mascara]
            grupos_busqueda = grupos_train[mascara]
        else:
            X_busqueda, y_busqueda = X_train, y_train
            grupos_busqueda = grupos_train

        pipeline = construir_pipeline(corrida.modelo, corrida.balanceo,
                                      roles, semilla,
                                      corrida.razon_desbalance)

        inicio = time.perf_counter()
        resultado = optimizar(corrida.optimizador, pipeline, espacio,
                              X_busqueda, y_busqueda, grupos_busqueda,
                              interno, metrica, corrida.presupuesto, semilla)
        t_busqueda += time.perf_counter() - inicio

        pipeline = construir_pipeline(corrida.modelo, corrida.balanceo,
                                      roles, semilla,
                                      corrida.razon_desbalance)
        pipeline.set_params(**resultado["mejores_parametros"])

        inicio = time.perf_counter()
        pipeline.fit(X_train, y_train)
        t_ajuste += time.perf_counter() - inicio

        inicio = time.perf_counter()
        probabilidades = pipeline.predict_proba(X_val)[:, 1]
        t_inferencia += time.perf_counter() - inicio

        por_fold.append(calcular_metricas(y_val, probabilidades))
        elegidos.append(resultado["mejores_parametros"])
        historiales.append(resultado["historial"])
        diversidades.append(resultado["diversidad"])

    fila = {
        "modelo": corrida.modelo,
        "balanceo": corrida.balanceo,
        "optimizador": corrida.optimizador,
        "presupuesto": corrida.presupuesto,
        "folds_externos": folds_externos,
        "folds_internos": folds_internos,
        "fraccion_busqueda": fraccion_busqueda,
    }
    for metrica_nombre in METRICAS:
        valores = [f[metrica_nombre] for f in por_fold]
        fila[f"{metrica_nombre}_media"] = float(np.mean(valores))
        fila[f"{metrica_nombre}_sd"] = float(np.std(valores, ddof=1))
    fila.update({
        "tiempo_busqueda_s": t_busqueda,
        "tiempo_ajuste_s": t_ajuste,
        "tiempo_inferencia_s": t_inferencia,
        "hiperparametros_por_fold": json.dumps(elegidos, default=str),
        "curvas_anytime": json.dumps(historiales),
        "curvas_diversidad": json.dumps(diversidades),
        "auc_pr_por_fold": json.dumps([f["auc_pr"] for f in por_fold]),
        "auc_roc_por_fold": json.dumps([f["auc_roc"] for f in por_fold]),
    })
    return fila


def ejecutar_experimento(corridas, X, y, grupos, roles, ruta_tabla,
                         verbose=True, **kwargs):
    """Ejecuta un conjunto de corridas con puntos de control en disco.

    Cada corrida se escribe en la tabla maestra en cuanto termina, y las ya
    presentes se omiten al reanudar. Con 112 combinaciones, esto evita perder
    horas de cómputo por una interrupción y permite ejecutar el experimento
    en varias sesiones.

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
            fila = ejecutar_corrida(corrida, X, y, grupos, roles, **kwargs)
            fila["tiempo_total_s"] = time.perf_counter() - inicio
            fila["estado"] = "completada"
        except (ValueError, ImportError) as exc:
            fila = {
                "modelo": corrida.modelo, "balanceo": corrida.balanceo,
                "optimizador": corrida.optimizador,
                "estado": "no aplicable", "detalle": str(exc),
            }
        tabla = pd.concat([tabla, pd.DataFrame([fila])], ignore_index=True)
        tabla.to_csv(ruta_tabla, index=False)
        if verbose:
            estado = fila["estado"]
            detalle = (f"AUC-PR {fila['auc_pr_media']:.4f} "
                       f"± {fila['auc_pr_sd']:.4f} "
                       f"en {fila['tiempo_total_s']:.1f} s"
                       if estado == "completada" else fila.get("detalle", ""))
            print(f"[{estado}] {corrida.identificador}: {detalle}")

    return tabla
