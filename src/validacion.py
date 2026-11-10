"""
Validación cruzada anidada y agrupada por paciente, y ejecutor del
experimento con puntos de control en disco.

Forma parte de ``src/``, la versión modular del motor del experimento. Las
funciones se trasladaron sin cambios de lógica desde ``docs/experimento.py``,
que se conserva congelado para reproducir el Entregable 2; la equivalencia
entre ambos se verifica en ``tests/test_equivalencia_e2.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from . import configuracion as cfg
from .balanceo import construir_pipeline
from .configuracion import SEMILLA, CombinacionNoAplicable
from .espacios import espacio_busqueda, presupuesto_modelo, rejilla_hiperparametros
from .metricas import METRICA_BUSQUEDA, METRICAS, calcular_metricas
from .modelos import PARALELO_INTERNO
from .optimizacion import optimizar


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


def _procesar_fold(i, indices_train, indices_val, corrida, X, y, grupos, roles,
                   folds_internos, metrica, presupuesto, fraccion_busqueda,
                   multifidelidad, semilla, n_jobs_busqueda):
    """Búsqueda, ajuste final y evaluación de un fold externo.

    Está a nivel de módulo, y no dentro de :func:`ejecutar_corrida`, para
    que ``joblib`` pueda enviarla a otro proceso cuando los folds externos se
    ejecutan en paralelo.

    El generador aleatorio de la submuestra de búsqueda se deriva de la
    semilla y del número de fold (``semilla + i``), no se comparte entre
    folds: así cada fold obtiene siempre la misma submuestra, se ejecute en
    serie o en paralelo y en el orden que sea.

    Returns
    -------
    dict
        Métricas del fold, hiperparámetros elegidos, curvas, conteos y
        tiempos por fase.
    """
    t_fold = time.perf_counter()
    X_train, X_val = X.iloc[indices_train], X.iloc[indices_val]
    y_train, y_val = y.iloc[indices_train], y.iloc[indices_val]
    grupos_train = grupos.iloc[indices_train]

    if fraccion_busqueda < 1.0:
        aleatorio = np.random.default_rng(semilla + i)
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

    pipeline = construir_pipeline(corrida.modelo, corrida.balanceo, roles,
                                  semilla, corrida.razon_desbalance)

    # El reparto agrupado no cambia entre configuraciones, así que se calcula
    # una vez por fold externo en lugar de una vez por evaluación.
    interno = StratifiedGroupKFold(n_splits=folds_internos, shuffle=True,
                                   random_state=semilla)
    particiones_internas = list(
        interno.split(X_busqueda, y_busqueda, groups=grupos_busqueda))

    inicio = time.perf_counter()
    resultado = optimizar(
        corrida.optimizador, pipeline,
        rejilla_hiperparametros(corrida.modelo),
        espacio_busqueda(corrida.modelo), X_busqueda, y_busqueda,
        particiones_internas, metrica, presupuesto, semilla=semilla + i,
        paralelo_interno=corrida.modelo in PARALELO_INTERNO,
        multifidelidad=multifidelidad, n_jobs=n_jobs_busqueda)
    t_busqueda = time.perf_counter() - inicio

    pipeline = clone(pipeline).set_params(**resultado["mejores_parametros"])
    inicio = time.perf_counter()
    pipeline.fit(X_train, y_train)
    t_ajuste = time.perf_counter() - inicio

    inicio = time.perf_counter()
    probabilidades = pipeline.predict_proba(X_val)[:, 1]
    t_inferencia = time.perf_counter() - inicio

    return {
        "metricas": calcular_metricas(y_val, probabilidades),
        "mejores_parametros": resultado["mejores_parametros"],
        "historial": resultado["historial"],
        "diversidad": resultado["diversidad"],
        "n_evaluaciones": len(resultado["evaluaciones"]),
        "n_ajustes": resultado["ajustes"] + 1,
        "t_busqueda": t_busqueda,
        "t_ajuste": t_ajuste,
        "t_inferencia": t_inferencia,
        "t_fold": time.perf_counter() - t_fold,
    }


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

    Paralelismo entre folds externos
    --------------------------------
    En las corridas de Optuna con modelos secuenciales (los que no están en
    :data:`PARALELO_INTERNO`), los folds externos se ejecutan a la vez, con
    un tope de :data:`N_JOBS` procesos, y la búsqueda dentro de cada fold va
    en un solo proceso. Optuna no puede repartir sus configuraciones, porque
    cada propuesta depende de las anteriores, pero los folds externos son
    independientes entre sí: ejecutarlos a la vez da exactamente las mismas
    métricas que en serie. En el resto de corridas el paralelismo ya está en
    otro nivel (configuraciones, o hilos dentro del modelo) y los folds
    externos van en serie. La tabla lo registra en ``folds_en_paralelo``.

    Tiempos que se registran
    ------------------------
    ``tiempo_busqueda_s``, ``tiempo_ajuste_s`` y ``tiempo_inferencia_s`` son
    la suma sobre los folds: miden el **trabajo** de cada fase. Con folds en
    serie, esa suma coincide con el tiempo transcurrido; con folds en
    paralelo, lo supera. ``tiempo_trabajo_s`` suma la duración de los folds y
    ``tiempo_real_s`` es el tiempo de reloj de la corrida, el que cuesta en
    la práctica. Las comparaciones de costo deben declarar cuál de los dos
    usan.

    Con ``verbose=True`` imprime una línea por fold externo con su duración
    y su AUC-PR. No altera ningún resultado.

    Returns
    -------
    dict
        Una fila de la tabla maestra.
    """
    t_corrida = time.perf_counter()
    metrica = metrica or METRICA_BUSQUEDA
    presupuesto = corrida.presupuesto or presupuesto_modelo(corrida.modelo)

    # Construir el pipeline una vez antes del bucle valida la combinación:
    # una combinación no aplicable se detecta aquí, antes de imprimir nada.
    construir_pipeline(corrida.modelo, corrida.balanceo, roles, semilla,
                       corrida.razon_desbalance)

    externo = StratifiedGroupKFold(n_splits=folds_externos, shuffle=True,
                                   random_state=semilla)
    divisiones = list(externo.split(X, y, groups=grupos))

    folds_en_paralelo = (corrida.optimizador == "optuna"
                         and corrida.modelo not in PARALELO_INTERNO
                         and cfg.N_JOBS != 1)
    comunes = dict(corrida=corrida, X=X, y=y, grupos=grupos, roles=roles,
                   folds_internos=folds_internos, metrica=metrica,
                   presupuesto=presupuesto,
                   fraccion_busqueda=fraccion_busqueda,
                   multifidelidad=multifidelidad, semilla=semilla)

    if folds_en_paralelo:
        if verbose:
            print(f"  {folds_externos} folds externos en paralelo "
                  f"(hasta {cfg.N_JOBS if cfg.N_JOBS > 0 else 'todos los'} "
                  "procesos)...", flush=True)
        resultados = Parallel(n_jobs=cfg.N_JOBS)(
            delayed(_procesar_fold)(i, ent, val, n_jobs_busqueda=1,
                                    **comunes)
            for i, (ent, val) in enumerate(divisiones, 1))
        if verbose:
            for i, r in enumerate(resultados, 1):
                print(f"  fold externo {i}/{folds_externos}: "
                      f"{r['t_fold']:.1f} s (auc_pr "
                      f"{r['metricas']['auc_pr']:.4f})", flush=True)
    else:
        resultados = []
        for i, (ent, val) in enumerate(divisiones, 1):
            if verbose:
                print(f"  fold externo {i}/{folds_externos}...", end=" ",
                      flush=True)
            r = _procesar_fold(i, ent, val, n_jobs_busqueda=cfg.N_JOBS,
                               **comunes)
            if verbose:
                print(f"listo en {r['t_fold']:.1f} s "
                      f"(auc_pr {r['metricas']['auc_pr']:.4f})", flush=True)
            resultados.append(r)

    t_busqueda = sum(r["t_busqueda"] for r in resultados)
    n_evaluaciones = sum(r["n_evaluaciones"] for r in resultados)
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
        # Condiciones de ejecución: no afectan a las métricas, pero sí a los
        # tiempos, así que se registran para que las comparaciones de costo
        # solo mezclen corridas medidas en las mismas condiciones.
        "n_jobs": cfg.N_JOBS,
        "folds_en_paralelo": folds_en_paralelo,
        "n_evaluaciones": n_evaluaciones,
        "n_ajustes": sum(r["n_ajustes"] for r in resultados),
    }
    for nombre in METRICAS:
        valores = [r["metricas"][nombre] for r in resultados]
        fila[f"{nombre}_media"] = float(np.mean(valores))
        fila[f"{nombre}_sd"] = float(np.std(valores, ddof=1))
        fila[f"{nombre}_por_fold"] = json.dumps(valores)
    fila.update({
        "tiempo_busqueda_s": t_busqueda,
        "tiempo_ajuste_s": sum(r["t_ajuste"] for r in resultados),
        "tiempo_inferencia_s": sum(r["t_inferencia"] for r in resultados),
        "tiempo_por_evaluacion_s": t_busqueda / max(n_evaluaciones, 1),
        "tiempo_trabajo_s": sum(r["t_fold"] for r in resultados),
        "tiempo_real_s": time.perf_counter() - t_corrida,
        "hiperparametros_por_fold": json.dumps(
            [r["mejores_parametros"] for r in resultados], default=str),
        "curvas_anytime": json.dumps([r["historial"] for r in resultados]),
        "curvas_diversidad": json.dumps(
            [r["diversidad"] for r in resultados]),
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
                "n_jobs": cfg.N_JOBS,
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
