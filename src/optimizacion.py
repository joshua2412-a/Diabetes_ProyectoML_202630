"""
Los cuatro optimizadores de hiperparámetros: Grid Search, Random Search,
optimización bayesiana (Optuna) y algoritmo genético.

Forma parte de ``src/``, la versión modular del motor del experimento. Las
funciones se trasladaron sin cambios de lógica desde ``docs/experimento.py``,
que se conserva congelado para reproducir el Entregable 2; la equivalencia
entre ambos se verifica en ``tests/test_equivalencia_e2.py``.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, cross_val_score

from . import configuracion as cfg
from .configuracion import SEMILLA, requiere
from .espacios import discretizar, muestrear

#: Los cuatro optimizadores de hiperparámetros.
OPTIMIZADORES = ["grid", "random", "optuna", "deap"]


def puntuar(pipeline, parametros, X, y, particiones, metrica):
    """Puntaje medio de validación cruzada de una configuración.

    Recibe las particiones ya calculadas (una lista de pares de índices) en
    lugar de un particionador: el reparto agrupado es costoso y no cambia
    entre configuraciones, de modo que se calcula una sola vez por fold
    externo (véase el perfilamiento del capítulo 6).
    """
    candidato = clone(pipeline).set_params(**parametros)
    return float(np.mean(cross_val_score(
        candidato, X, y, cv=particiones, scoring=metrica, n_jobs=1,
        error_score="raise")))


def _puntuar_con_poda(pipeline, parametros, X, y, particiones, metrica,
                      intento, optuna):
    """Evalúa fold a fold e informa a Optuna para que pueda podar.

    La fidelidad es el número de folds internos evaluados: tras cada fold se
    reporta la media acumulada y el podador decide si la configuración sigue.
    """
    from sklearn.metrics import get_scorer
    evaluador = get_scorer(metrica)
    puntajes = []
    for paso, (i_ent, i_val) in enumerate(particiones, 1):
        candidato = clone(pipeline).set_params(**parametros)
        candidato.fit(X.iloc[i_ent], y.iloc[i_ent])
        puntajes.append(evaluador(candidato, X.iloc[i_val], y.iloc[i_val]))
        intento.report(float(np.mean(puntajes)), paso)
        if intento.should_prune():
            intento.set_user_attr("folds_evaluados", paso)
            raise optuna.TrialPruned()
    intento.set_user_attr("folds_evaluados", len(puntajes))
    return float(np.mean(puntajes))


def optimizar(estrategia, pipeline, rejilla, espacio, X, y, particiones,
              metrica, presupuesto, semilla=SEMILLA, paralelo_interno=False,
              multifidelidad=True, n_jobs=None):
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

    ``particiones`` es la lista de pares de índices que produce el
    particionador interno, calculada una sola vez fuera de esta función: el
    reparto agrupado por paciente cuesta más que el propio ajuste de los
    modelos lineales, y recalcularlo por configuración multiplicaba ese costo
    por el presupuesto (véase el perfilamiento del capítulo 6).

    Paralelización: si el modelo no paraleliza internamente, las
    configuraciones de Grid, Random y de cada generación del genético se
    evalúan en paralelo. Optuna es secuencial por naturaleza (cada propuesta
    depende de las anteriores); su paralelismo se obtiene un nivel más
    arriba, ejecutando a la vez los folds externos (véase
    :func:`ejecutar_corrida`). ``n_jobs`` fija el número de procesos de esta
    búsqueda; por defecto, :data:`N_JOBS`. Cuando la búsqueda corre dentro de
    un fold externo que ya se ejecuta en paralelo, se pasa ``n_jobs=1`` para
    no anidar paralelismo.

    Returns
    -------
    dict
        Mejores parámetros y puntaje, curva anytime (mejor puntaje tras cada
        evaluación), evaluaciones, diversidad genética y número de ajustes
        de pipeline consumidos.
    """
    aleatorio = np.random.default_rng(semilla)
    n_folds = len(particiones)
    procesos = cfg.N_JOBS if n_jobs is None else n_jobs
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
        if not paralelo_interno and procesos != 1 and len(lista) > 1:
            puntajes = Parallel(n_jobs=procesos)(
                delayed(puntuar)(pipeline, p, X, y, particiones, metrica)
                for p in lista)
        else:
            puntajes = [puntuar(pipeline, p, X, y, particiones, metrica)
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
        candidatos = [{c: muestrear(d, aleatorio) for c, d in espacio.items()}
                      for _ in range(presupuesto)]
        puntajes = evaluar_lote(candidatos)
        mejor = int(np.argmax(puntajes))
        mejores, mejor_puntaje = candidatos[mejor], puntajes[mejor]

    elif estrategia == "optuna":
        optuna = requiere("optuna")
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
                                                particiones, metrica,
                                                intento, optuna)
                except optuna.TrialPruned:
                    # Configuración podada: consume los folds que alcanzó a
                    # evaluar, pero no compite por el mejor puntaje.
                    registrar(parametros, float("nan"),
                              intento.user_attrs.get("folds_evaluados", 1))
                    raise
            else:
                candidato = clone(pipeline).set_params(**parametros)
                puntaje = float(np.mean(cross_val_score(
                    candidato, X, y, cv=particiones, scoring=metrica,
                    n_jobs=1 if paralelo_interno else procesos,
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
    opciones = [discretizar(espacio[c]) for c in claves]
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
