"""
Correcciones del Entregable 2 (capítulo 12).

Reúne el código del capítulo: el estudio de convergencia del presupuesto de
búsqueda, los tamaños del efecto paramétricos (d de Cohen) y las
verificaciones automáticas de la auditoría de fuga.

El módulo **no modifica** ``experimento.py``: lo reutiliza. El estudio de
convergencia tiene que evaluar exactamente el mismo código, las mismas
particiones y la misma semilla que produjeron las 112 corridas; si el código
cambiara, una diferencia entre presupuestos podría deberse al cambio de
código y no al presupuesto.

Puntos de control
-----------------
Todo cálculo costoso escribe en ``results/`` a medida que avanza y, al
reanudar, omite lo que ya está en disco. El capítulo se puede interrumpir y
volver a lanzar sin perder trabajo, y en modo de lectura (sin nada que
calcular) solo lee esos archivos, de modo que se ejecuta en segundos.

scikit-learn, joblib y el propio motor se importan dentro de las funciones
que los necesitan: la lectura de resultados y los tamaños del efecto solo
dependen de numpy y pandas.
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

#: Presupuesto máximo del estudio de convergencia, en configuraciones por
#: fold externo. Con 60 muestras independientes, la probabilidad de que al
#: menos una caiga en el 5 % mejor del espacio de búsqueda es
#: 1 - 0.95**60 ≈ 0.954 (Bergstra y Bengio, 2012), con independencia del
#: número de dimensiones.
PRESUPUESTO_MAXIMO = 60

#: Presupuestos en los que se corta la secuencia de Random Search para trazar
#: la curva: los dos del Entregable 2 (8 y 12) y cuatro mayores.
CORTES = (8, 12, 20, 30, 45, 60)

#: Configuraciones que se evalúan entre dos escrituras en disco. Una
#: interrupción pierde como mucho un lote.
TAMANO_LOTE = 12

#: Región de equivalencia práctica para la ganancia de AUC-PR, fijada antes
#: de ver los resultados (sección 12.2.1): la mitad de la menor desviación
#: entre folds de la mejor configuración de cada modelo en el Entregable 2.
ROPE_AUC_PR = 0.005

#: Las mismas métricas que registra el motor (``experimento.METRICAS``),
#: repetidas aquí para no importar scikit-learn al solo leer resultados.
METRICAS = ["auc_roc", "auc_pr", "exactitud", "precision", "recall", "f1",
            "exactitud_balanceada", "brier"]


# ==========================================================================
# Utilidades de E/S con puntos de control
# ==========================================================================

def _leer(ruta):
    """Lee un CSV de resultados; devuelve un DataFrame vacío si no existe."""
    ruta = Path(ruta)
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


def _anexar(ruta, filas):
    """Añade filas a un CSV y crea el encabezado si el archivo no existe.

    Se escribe tras cada lote, no al final: es lo que permite reanudar.
    """
    if not filas:
        return
    ruta = Path(ruta)
    pd.DataFrame(filas).to_csv(ruta, mode="a", header=not ruta.exists(),
                               index=False)


def _filtrar(datos, modelo, balanceo, fold):
    """Filas de una combinación y un fold externo."""
    if datos.empty:
        return datos
    return datos[(datos["modelo"] == modelo) & (datos["balanceo"] == balanceo)
                 & (datos["fold"] == fold)]


def calcular_o_leer(ruta, calcular, recalcular=False):
    """Lee un resultado guardado o lo calcula y lo guarda.

    Parameters
    ----------
    ruta : str or pathlib.Path
        CSV donde vive el resultado.
    calcular : callable
        Función sin argumentos que devuelve un DataFrame. Solo se llama si
        el archivo no existe o si ``recalcular`` es verdadero.
    recalcular : bool, default False

    Returns
    -------
    pandas.DataFrame
    """
    ruta = Path(ruta)
    if ruta.exists() and not recalcular:
        return pd.read_csv(ruta)
    datos = calcular()
    datos.to_csv(ruta, index=False)
    return datos


# ==========================================================================
# Réplica exacta de las piezas aleatorias del motor
# ==========================================================================

def candidatos_aleatorios(modelo, presupuesto, semilla):
    """Configuraciones de Random Search, idénticas a las del motor.

    ``experimento.optimizar`` crea un generador con la semilla del fold y
    muestrea las configuraciones una tras otra, en el orden de las claves del
    espacio. Por eso las ``k`` primeras de una secuencia de 60 son exactamente
    las ``k`` que habría evaluado una búsqueda con presupuesto ``k``: los
    presupuestos menores son prefijos de la misma secuencia.

    Parameters
    ----------
    modelo : str
    presupuesto : int
    semilla : int
        Semilla del fold externo (semilla global + número de fold), la misma
        que usa el motor.

    Returns
    -------
    list of dict
    """
    import experimento as ex
    aleatorio = np.random.default_rng(semilla)
    espacio = ex.espacio_busqueda(modelo)
    return [{c: ex._muestrear(d, aleatorio) for c, d in espacio.items()}
            for _ in range(presupuesto)]


def datos_busqueda(X_train, y_train, grupos_train, fraccion, semilla_fold):
    """Submuestra de pacientes para la búsqueda, idéntica a la del motor.

    Replica la primera parte de ``experimento._procesar_fold``: con
    ``fraccion`` < 1 la búsqueda usa una fracción de los **pacientes** del
    fold externo de entrenamiento, elegida con un generador propio del fold.

    Returns
    -------
    tuple
        ``(X, y, grupos)`` de la búsqueda.
    """
    if fraccion < 1.0:
        aleatorio = np.random.default_rng(semilla_fold)
        pacientes = grupos_train.drop_duplicates()
        elegidos = aleatorio.choice(
            pacientes.to_numpy(),
            size=max(2, int(len(pacientes) * fraccion)),
            replace=False)
        mascara = grupos_train.isin(elegidos).to_numpy()
        return X_train[mascara], y_train[mascara], grupos_train[mascara]
    return X_train, y_train, grupos_train


def particiones_externas(X, y, grupos, folds_externos=5, semilla=None):
    """Particiones del bucle externo, las mismas que usa el motor."""
    import experimento as ex
    from sklearn.model_selection import StratifiedGroupKFold
    semilla = ex.SEMILLA if semilla is None else semilla
    externo = StratifiedGroupKFold(n_splits=folds_externos, shuffle=True,
                                   random_state=semilla)
    return list(externo.split(X, y, groups=grupos))


# ==========================================================================
# Estudio de convergencia del presupuesto
# ==========================================================================

def ejecutar_convergencia(modelo, balanceo, X, y, grupos, roles,
                          ruta_evaluaciones, ruta_reajustes,
                          fraccion_busqueda=1.0, razon_desbalance=None,
                          presupuesto=PRESUPUESTO_MAXIMO, cortes=CORTES,
                          folds_externos=5, folds_internos=3, metrica=None,
                          semilla=None, verbose=True):
    """Random Search con presupuesto amplio y reajuste en cada corte.

    Para cada fold externo:

    1. Evalúa las ``presupuesto`` configuraciones de Random Search con la
       validación interna, exactamente como el motor, y guarda el puntaje
       interno de cada una en ``ruta_evaluaciones``.
    2. Para cada corte ``k`` toma la mejor de las ``k`` primeras (la que
       habría elegido una búsqueda con presupuesto ``k``), la reajusta con
       todo el fold externo de entrenamiento y la mide en el fold de
       evaluación. Cada configuración elegida se reajusta una sola vez,
       aunque gane en varios cortes, y se guarda en ``ruta_reajustes``.

    Ambos archivos se escriben a medida que avanza el cálculo y lo que ya
    está en ellos se omite al reanudar.

    Parameters
    ----------
    modelo, balanceo : str
    X, y, grupos : pandas objects
        Conjunto de entrenamiento completo, como en el capítulo 8.
    roles : dict
    ruta_evaluaciones, ruta_reajustes : str or pathlib.Path
    fraccion_busqueda : float
        La misma del Entregable 2 para ese modelo.
    razon_desbalance : float, optional
        Necesaria para ``class_weight`` en XGBoost.
    presupuesto : int
    cortes : sequence of int
    folds_externos, folds_internos : int
    metrica : str, optional
        Por defecto, ``experimento.METRICA_BUSQUEDA``.
    semilla : int, optional
        Por defecto, ``experimento.SEMILLA``.
    verbose : bool
    """
    import experimento as ex
    from joblib import Parallel, delayed
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedGroupKFold

    metrica = metrica or ex.METRICA_BUSQUEDA
    semilla = ex.SEMILLA if semilla is None else semilla
    cortes = [k for k in cortes if k <= presupuesto]
    divisiones = particiones_externas(X, y, grupos, folds_externos, semilla)

    # El pipeline se construye con la semilla global, como en el motor.
    pipeline_base = ex.construir_pipeline(modelo, balanceo, roles, semilla,
                                          razon_desbalance)
    # Misma regla de paralelismo que el motor: si el modelo ya paraleliza
    # internamente, las configuraciones van en serie.
    paralelo = modelo not in ex.PARALELO_INTERNO and ex.N_JOBS != 1

    for i, (indices_train, indices_val) in enumerate(divisiones, 1):
        t_fold = time.perf_counter()
        X_train, X_val = X.iloc[indices_train], X.iloc[indices_val]
        y_train, y_val = y.iloc[indices_train], y.iloc[indices_val]
        grupos_train = grupos.iloc[indices_train]
        candidatos = candidatos_aleatorios(modelo, presupuesto, semilla + i)

        # --- 1. Búsqueda: puntaje interno de cada configuración ----------
        previas = _filtrar(_leer(ruta_evaluaciones), modelo, balanceo, i)
        hechas = set(previas["indice"]) if len(previas) else set()
        pendientes = [j for j in range(presupuesto) if j not in hechas]

        if pendientes:
            X_b, y_b, g_b = datos_busqueda(X_train, y_train, grupos_train,
                                           fraccion_busqueda, semilla + i)
            # Las particiones internas se calculan una vez por fold, como en
            # el motor: el reparto agrupado no cambia entre configuraciones.
            interno = StratifiedGroupKFold(n_splits=folds_internos,
                                           shuffle=True, random_state=semilla)
            particiones = list(interno.split(X_b, y_b, groups=g_b))

            for inicio in range(0, len(pendientes), TAMANO_LOTE):
                lote = pendientes[inicio:inicio + TAMANO_LOTE]
                t_lote = time.perf_counter()
                if paralelo:
                    puntajes = Parallel(n_jobs=ex.N_JOBS)(
                        delayed(ex._puntuar)(pipeline_base, candidatos[j],
                                             X_b, y_b, particiones, metrica)
                        for j in lote)
                else:
                    puntajes = [ex._puntuar(pipeline_base, candidatos[j], X_b,
                                            y_b, particiones, metrica)
                                for j in lote]
                por_evaluacion = (time.perf_counter() - t_lote) / len(lote)
                _anexar(ruta_evaluaciones, [{
                    "modelo": modelo, "balanceo": balanceo, "fold": i,
                    "indice": j,
                    "parametros": json.dumps(candidatos[j], default=str),
                    "puntaje_interno": float(s),
                    "tiempo_s": por_evaluacion,
                    # Condición de ejecución: no afecta a los puntajes, pero
                    # sí a los tiempos (igual que en la tabla maestra).
                    "n_jobs": ex.N_JOBS,
                } for j, s in zip(lote, puntajes)])
                if verbose:
                    hechas_ahora = len(hechas) + inicio + len(lote)
                    print(f"  fold {i}: {hechas_ahora}/{presupuesto} "
                          f"evaluaciones ({por_evaluacion:.1f} s cada una)",
                          flush=True)

        evaluaciones = (_filtrar(_leer(ruta_evaluaciones), modelo, balanceo, i)
                        .drop_duplicates("indice").set_index("indice"))
        puntajes = (evaluaciones["puntaje_interno"]
                    .reindex(range(presupuesto)).to_numpy())

        # --- 2. Reajuste de la configuración elegida en cada corte --------
        # np.argmax devuelve el primer máximo, la misma regla del motor.
        elegidos = sorted({int(np.argmax(puntajes[:k])) for k in cortes})
        previos = _filtrar(_leer(ruta_reajustes), modelo, balanceo, i)
        ya_reajustados = set(previos["indice"]) if len(previos) else set()

        for j in elegidos:
            if j in ya_reajustados:
                continue
            final = clone(pipeline_base).set_params(**candidatos[j])
            t0 = time.perf_counter()
            final.fit(X_train, y_train)
            t_ajuste = time.perf_counter() - t0
            t0 = time.perf_counter()
            probabilidades = final.predict_proba(X_val)[:, 1]
            t_inferencia = time.perf_counter() - t0
            _anexar(ruta_reajustes, [{
                "modelo": modelo, "balanceo": balanceo, "fold": i,
                "indice": j,
                "parametros": json.dumps(candidatos[j], default=str),
                "puntaje_interno": float(puntajes[j]),
                **ex.calcular_metricas(y_val, probabilidades),
                "tiempo_ajuste_s": t_ajuste,
                "tiempo_inferencia_s": t_inferencia,
                "n_jobs": ex.N_JOBS,
            }])

        if verbose:
            print(f"  fold {i} listo en {time.perf_counter() - t_fold:.1f} s",
                  flush=True)


def _prefijo_contiguo(puntajes_por_indice):
    """Puntajes de los índices 0, 1, 2, ... hasta el primer hueco.

    Con un cálculo interrumpido puede faltar parte de la secuencia; solo el
    prefijo sin huecos representa búsquedas con presupuesto conocido.
    """
    n = 0
    while n in puntajes_por_indice.index:
        n += 1
    return puntajes_por_indice.reindex(range(n)).to_numpy()


def tabla_convergencia(ruta_evaluaciones, ruta_reajustes, cortes=CORTES):
    """Desempeño externo de la configuración elegida en cada corte.

    Se reconstruye a partir de los dos archivos de puntos de control, de
    modo que no requiere volver a ajustar nada.

    Returns
    -------
    pandas.DataFrame
        Una fila por modelo, balanceo, fold y presupuesto, con el índice de
        la configuración elegida, su puntaje interno y las métricas del fold
        externo.
    """
    evaluaciones, reajustes = _leer(ruta_evaluaciones), _leer(ruta_reajustes)
    if evaluaciones.empty or reajustes.empty:
        return pd.DataFrame()

    filas = []
    for (modelo, balanceo, fold), grupo in evaluaciones.groupby(
            ["modelo", "balanceo", "fold"]):
        p = _prefijo_contiguo(grupo.drop_duplicates("indice")
                              .set_index("indice")["puntaje_interno"])
        reaj = (_filtrar(reajustes, modelo, balanceo, fold)
                .drop_duplicates("indice").set_index("indice"))
        for k in cortes:
            if k > len(p):
                continue
            j = int(np.argmax(p[:k]))
            if j not in reaj.index:
                continue
            fila = {"modelo": modelo, "balanceo": balanceo, "fold": int(fold),
                    "presupuesto": k, "indice_elegido": j,
                    "mejor_puntaje_interno": float(p[j])}
            fila.update({m: float(reaj.at[j, m]) for m in METRICAS
                         if m in reaj.columns})
            filas.append(fila)
    return pd.DataFrame(filas)


def curva_anytime(ruta_evaluaciones):
    """Mejor puntaje interno encontrado tras cada evaluación.

    Returns
    -------
    pandas.DataFrame
        Columnas ``modelo``, ``balanceo``, ``fold``, ``evaluacion`` (1, 2,
        ...) y ``mejor_hasta_ahora``.
    """
    evaluaciones = _leer(ruta_evaluaciones)
    if evaluaciones.empty:
        return pd.DataFrame()
    partes = []
    for (modelo, balanceo, fold), grupo in evaluaciones.groupby(
            ["modelo", "balanceo", "fold"]):
        p = _prefijo_contiguo(grupo.drop_duplicates("indice")
                              .set_index("indice")["puntaje_interno"])
        partes.append(pd.DataFrame({
            "modelo": modelo, "balanceo": balanceo, "fold": int(fold),
            "evaluacion": np.arange(1, len(p) + 1),
            "mejor_hasta_ahora": np.maximum.accumulate(p),
        }))
    return pd.concat(partes, ignore_index=True)


def progreso(ruta_evaluaciones, ruta_reajustes, combinaciones,
             presupuesto=PRESUPUESTO_MAXIMO, cortes=CORTES, folds_externos=5):
    """Cuánto del estudio está calculado, por combinación.

    Parameters
    ----------
    combinaciones : iterable of tuple
        Pares ``(modelo, balanceo)`` del estudio.

    Returns
    -------
    pandas.DataFrame
    """
    evaluaciones = _leer(ruta_evaluaciones)
    convergencia = tabla_convergencia(ruta_evaluaciones, ruta_reajustes,
                                      cortes)
    n_cortes = len([k for k in cortes if k <= presupuesto])
    filas = []
    for modelo, balanceo in combinaciones:
        hechas = 0
        if not evaluaciones.empty:
            hechas = len(evaluaciones[(evaluaciones["modelo"] == modelo)
                                      & (evaluaciones["balanceo"] == balanceo)]
                         .drop_duplicates(["fold", "indice"]))
        completos = 0
        if not convergencia.empty:
            propios = convergencia[(convergencia["modelo"] == modelo)
                                   & (convergencia["balanceo"] == balanceo)]
            completos = int((propios.groupby("fold").size()
                             == n_cortes).sum())
        filas.append({
            "modelo": modelo, "balanceo": balanceo,
            "evaluaciones": f"{hechas}/{presupuesto * folds_externos}",
            "folds completos": f"{completos}/{folds_externos}",
            "estado": ("Completo" if completos == folds_externos
                       else "Sin iniciar" if hechas == 0 else "En curso"),
        })
    return pd.DataFrame(filas)


def verificar_reproduccion(convergencia, maestra, tolerancia=1e-6):
    """Compara el corte del Entregable 2 con su corrida de Random Search.

    El corte igual al presupuesto del Entregable 2 debe reproducir, fold a
    fold, la corrida ``random`` de la misma combinación en la tabla maestra.
    Es la prueba de que el estudio usa el mismo protocolo.

    Parameters
    ----------
    convergencia : pandas.DataFrame
        Salida de :func:`tabla_convergencia`.
    maestra : pandas.DataFrame
        Tabla maestra del Entregable 2 (corridas completadas).
    tolerancia : float

    Returns
    -------
    pandas.DataFrame
    """
    filas = []
    for (modelo, balanceo), grupo in convergencia.groupby(["modelo",
                                                            "balanceo"]):
        referencia = maestra[(maestra["modelo"] == modelo)
                             & (maestra["balanceo"] == balanceo)
                             & (maestra["optimizador"] == "random")]
        if referencia.empty:
            continue
        referencia = referencia.iloc[0]
        k = int(referencia["presupuesto"])
        e2 = np.asarray(json.loads(referencia["auc_pr_por_fold"]))
        corte = grupo[grupo["presupuesto"] == k].sort_values("fold")
        if corte.empty:
            continue
        folds = corte["fold"].to_numpy()
        diferencias = np.abs(corte["auc_pr"].to_numpy() - e2[folds - 1])
        filas.append({
            "modelo": modelo, "balanceo": balanceo, "presupuesto": k,
            "folds comparados": len(folds),
            "AUC-PR E2 (random)": float(e2[folds - 1].mean()),
            "AUC-PR del corte": float(corte["auc_pr"].mean()),
            "máx. |diferencia| por fold": float(diferencias.max()),
            "reproduce": "Sí" if diferencias.max() <= tolerancia else "No",
        })
    return pd.DataFrame(filas)


def _t_critico(grados_libertad, confianza=0.95):
    """Valor crítico bilateral de la t de Student."""
    try:
        from scipy import stats
        return float(stats.t.ppf(0.5 + confianza / 2, grados_libertad))
    except ImportError:
        tabla = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
        return tabla.get(grados_libertad, 1.96)


def ganancias(convergencia, referencia):
    """Ganancia de AUC-PR de cada corte frente al presupuesto del E2.

    Parameters
    ----------
    convergencia : pandas.DataFrame
        Salida de :func:`tabla_convergencia`.
    referencia : dict
        Modelo a presupuesto del Entregable 2.

    Returns
    -------
    pandas.DataFrame
        Columnas ``modelo``, ``fold``, ``presupuesto`` y ``ganancia``.
    """
    partes = []
    for modelo, k_ref in referencia.items():
        grupo = convergencia[convergencia["modelo"] == modelo]
        if grupo.empty:
            continue
        tabla = grupo.pivot_table(index="fold", columns="presupuesto",
                                  values="auc_pr")
        if k_ref not in tabla.columns:
            continue
        delta = tabla.sub(tabla[k_ref], axis=0)
        largo = delta.reset_index().melt(id_vars="fold",
                                         var_name="presupuesto",
                                         value_name="ganancia").dropna()
        largo.insert(0, "modelo", modelo)
        partes.append(largo)
    return (pd.concat(partes, ignore_index=True) if partes
            else pd.DataFrame())


def resumen_ganancia(convergencia, referencia, maximo=PRESUPUESTO_MAXIMO,
                     rope=ROPE_AUC_PR, folds_externos=5):
    """Aplica la regla de decisión de la sección 12.2.1 a cada modelo.

    Parameters
    ----------
    convergencia : pandas.DataFrame
        Salida de :func:`tabla_convergencia`.
    referencia : dict
        Modelo a presupuesto del Entregable 2.
    maximo : int
    rope : float
        Ganancia media máxima con la que el presupuesto del E2 se considera
        suficiente.

    Returns
    -------
    pandas.DataFrame
    """
    filas = []
    for modelo, k_ref in referencia.items():
        grupo = convergencia[convergencia["modelo"] == modelo]
        if grupo.empty:
            continue
        tabla = grupo.pivot_table(index="fold", columns="presupuesto",
                                  values="auc_pr")
        if k_ref not in tabla.columns or maximo not in tabla.columns:
            continue
        tabla = tabla[[k_ref, maximo]].dropna()
        delta = tabla[maximo] - tabla[k_ref]
        n = len(delta)
        media = float(delta.mean())
        sd = float(delta.std(ddof=1)) if n > 1 else np.nan
        margen = (_t_critico(n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
        decision = "Suficiente" if media <= rope else "Insuficiente"
        if n < folds_externos:
            decision += f" (parcial: {n} folds)"
        filas.append({
            "modelo": modelo,
            "balanceo": grupo["balanceo"].iloc[0],
            "presupuesto E2": k_ref,
            "AUC-PR con presupuesto E2": float(tabla[k_ref].mean()),
            f"AUC-PR con {maximo}": float(tabla[maximo].mean()),
            "desviación entre folds (E2)": float(tabla[k_ref].std(ddof=1)),
            "ganancia media": media,
            "IC 95 % inferior": media - margen,
            "IC 95 % superior": media + margen,
            "decisión": decision,
        })
    return pd.DataFrame(filas)


# ==========================================================================
# Tamaño del efecto paramétrico
# ==========================================================================

def d_cohen(a, b, corregir=True):
    """d de Cohen con varianza combinada y corrección de Hedges.

    Trata las dos muestras como independientes, igual que la delta de Cliff
    del capítulo 11, para que ambas medidas sean comparables. Con muestras
    pequeñas, d sobrestima el efecto; el factor
    ``1 - 3 / (4 (n_a + n_b) - 9)`` lo corrige (g de Hedges).

    Returns
    -------
    float
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    varianza = (((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1))
                / (n_a + n_b - 2))
    if varianza == 0:
        return float("nan")
    d = (a.mean() - b.mean()) / np.sqrt(varianza)
    if corregir:
        d *= 1 - 3 / (4 * (n_a + n_b) - 9)
    return float(d)


def d_pareada(a, b):
    """d_z: media de las diferencias pareadas sobre su desviación.

    Es la versión coherente con el diseño pareado: los modelos se evalúan
    sobre los mismos folds externos, así que la variabilidad común a un fold
    se cancela en la diferencia.
    """
    diferencias = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = diferencias.std(ddof=1)
    return float(diferencias.mean() / sd) if sd > 0 else float("nan")


def magnitud_cohen(d):
    """Umbrales convencionales de Cohen: 0.2, 0.5 y 0.8."""
    if np.isnan(d):
        return "Indefinida"
    d = abs(d)
    if d < 0.2:
        return "Insignificante"
    if d < 0.5:
        return "Pequeño"
    if d < 0.8:
        return "Mediano"
    return "Grande"


def tabla_cohen(por_fold):
    """d de Cohen y d_z para todos los pares de columnas.

    Parameters
    ----------
    por_fold : pandas.DataFrame
        Filas: folds externos. Columnas: modelos.

    Returns
    -------
    pandas.DataFrame
    """
    filas = []
    for a, b in combinations(por_fold.columns, 2):
        d = d_cohen(por_fold[a], por_fold[b])
        dz = d_pareada(por_fold[a], por_fold[b])
        filas.append({
            "modelo A": a, "modelo B": b,
            "AUC-PR medio A": float(por_fold[a].mean()),
            "AUC-PR medio B": float(por_fold[b].mean()),
            "diferencia": float(por_fold[a].mean() - por_fold[b].mean()),
            "d de Cohen (Hedges)": d, "magnitud d": magnitud_cohen(d),
            "d_z pareada": dz, "magnitud d_z": magnitud_cohen(dz),
        })
    return pd.DataFrame(filas)


# ==========================================================================
# Auditoría de fuga
# ==========================================================================

def verificar_particiones(X, y, grupos, fracciones=(1.0, 0.25),
                          folds_externos=5, folds_internos=3, semilla=None):
    """Pacientes compartidos en cada partición de la validación anidada.

    Recalcula las particiones externas y, dentro de cada fold externo, las
    internas con cada fracción de búsqueda usada en el experimento, con el
    mismo código y la misma semilla que el motor.

    Returns
    -------
    pandas.DataFrame
        Una fila por partición; la columna ``pacientes compartidos`` debe
        ser cero en todas.
    """
    import experimento as ex
    from sklearn.model_selection import StratifiedGroupKFold
    semilla = ex.SEMILLA if semilla is None else semilla

    filas = []
    for i, (ent, val) in enumerate(
            particiones_externas(X, y, grupos, folds_externos, semilla), 1):
        g_ent, g_val = grupos.iloc[ent], grupos.iloc[val]
        filas.append({
            "nivel": "externo", "fracción de búsqueda": 1.0,
            "fold externo": i, "fold interno": 0,
            "encuentros entrenamiento": len(ent),
            "encuentros evaluación": len(val),
            "pacientes compartidos": len(set(g_ent) & set(g_val)),
            "tasa positiva evaluación": float(y.iloc[val].mean()),
        })
        for fraccion in fracciones:
            X_b, y_b, g_b = datos_busqueda(X.iloc[ent], y.iloc[ent], g_ent,
                                           fraccion, semilla + i)
            interno = StratifiedGroupKFold(n_splits=folds_internos,
                                           shuffle=True, random_state=semilla)
            for k, (e2, v2) in enumerate(interno.split(X_b, y_b, groups=g_b),
                                         1):
                filas.append({
                    "nivel": "interno", "fracción de búsqueda": fraccion,
                    "fold externo": i, "fold interno": k,
                    "encuentros entrenamiento": len(e2),
                    "encuentros evaluación": len(v2),
                    "pacientes compartidos": len(set(g_b.iloc[e2])
                                                 & set(g_b.iloc[v2])),
                    "tasa positiva evaluación": float(y_b.iloc[v2].mean()),
                })
    return pd.DataFrame(filas)


def verificar_pipelines(roles, razon_desbalance):
    """Estructura de los pipelines de todas las combinaciones aplicables.

    Comprueba que el preprocesamiento es la primera etapa del pipeline (se
    ajusta dentro de cada fold) y que el remuestreo, cuando existe, va
    después del preprocesador y dentro de un pipeline de imbalanced-learn,
    que solo lo aplica al ajustar y nunca al predecir.

    Returns
    -------
    pandas.DataFrame
    """
    import experimento as ex
    filas = []
    for modelo in ex.MODELOS:
        for balanceo in ex.BALANCEOS:
            try:
                pipeline = ex.construir_pipeline(modelo, balanceo, roles,
                                                 ex.SEMILLA, razon_desbalance)
            except ex.CombinacionNoAplicable:
                continue
            pasos = [nombre for nombre, _ in pipeline.steps]
            remuestrea = "balanceo" in pasos
            de_imblearn = type(pipeline).__module__.startswith("imblearn")
            filas.append({
                "modelo": modelo, "balanceo": balanceo,
                "etapas": " > ".join(pasos),
                "clase": (f"{type(pipeline).__module__.split('.')[0]}."
                          f"{type(pipeline).__name__}"),
                "preprocesamiento primero": pasos[0] == "preprocesamiento",
                "remuestreo solo al ajustar": (not remuestrea) or de_imblearn,
                "remuestreo tras el preprocesador": (
                    (not remuestrea)
                    or pasos.index("balanceo") > pasos.index("preprocesamiento")),
            })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Modularización y pruebas (sección 12.5)
# ---------------------------------------------------------------------------

#: Funciones que eran privadas en ``experimento.py`` y pasaron a ser públicas
#: en ``src/`` porque otro módulo las importa. Solo cambia el nombre.
RENOMBRADAS = {"_requiere": "requiere", "_muestrear": "muestrear",
               "_discretizar": "discretizar", "_puntuar": "puntuar"}


def _primera_linea(docstring):
    return (docstring or "").strip().splitlines()[0] if docstring else ""


def mapa_modulos(carpeta_src):
    """Tabla con cada módulo de ``src/``: propósito, líneas y qué define."""
    import ast

    filas = []
    for ruta in sorted(Path(carpeta_src).glob("*.py")):
        if ruta.stem == "__init__":
            continue
        arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        publicos = [n.name for n in arbol.body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                    and not n.name.startswith("_")]
        filas.append({
            "módulo": f"src/{ruta.name}",
            "propósito": _primera_linea(ast.get_docstring(arbol)),
            "líneas": len(ruta.read_text(encoding="utf-8").splitlines()),
            "funciones y clases públicas": ", ".join(publicos),
        })
    return pd.DataFrame(filas)


class _Normalizar:
    """Quita del código lo que la reorganización cambió sin cambiar la lógica.

    Dos cosas: las funciones privadas que pasaron a públicas
    (``RENOMBRADAS``) y el acceso a la configuración, que en ``src/`` se lee
    como ``cfg.N_JOBS`` para que un cambio en tiempo de ejecución llegue a
    todos los módulos.
    """

    def __new__(cls, arbol):
        import ast

        class _T(ast.NodeTransformer):
            def visit_Attribute(self, nodo):
                self.generic_visit(nodo)
                if isinstance(nodo.value, ast.Name) and nodo.value.id == "cfg":
                    return ast.Name(id=nodo.attr, ctx=nodo.ctx)
                return nodo

            def visit_Name(self, nodo):
                nodo.id = RENOMBRADAS.get(nodo.id, nodo.id)
                return nodo

            def visit_FunctionDef(self, nodo):
                self.generic_visit(nodo)
                nodo.name = RENOMBRADAS.get(nodo.name, nodo.name)
                return nodo

        return _T().visit(arbol)


def _definiciones(ruta):
    """Código normalizado, sin docstrings, de cada función, clase y constante."""
    import ast

    arbol = _Normalizar(ast.parse(Path(ruta).read_text(encoding="utf-8")))
    salida = {}
    for nodo in arbol.body:
        if isinstance(nodo, (ast.FunctionDef, ast.ClassDef)):
            cuerpo = nodo.body
            if (cuerpo and isinstance(cuerpo[0], ast.Expr)
                    and isinstance(cuerpo[0].value, ast.Constant)
                    and isinstance(cuerpo[0].value.value, str)):
                cuerpo = cuerpo[1:]
            for sub in ast.walk(ast.Module(body=cuerpo, type_ignores=[])):
                if (isinstance(sub, (ast.FunctionDef, ast.ClassDef))
                        and sub.body and isinstance(sub.body[0], ast.Expr)
                        and isinstance(sub.body[0].value, ast.Constant)
                        and isinstance(sub.body[0].value.value, str)):
                    sub.body = sub.body[1:] or [ast.Pass()]
            firma = (ast.unparse(nodo.args) if isinstance(nodo, ast.FunctionDef)
                     else "")
            salida[nodo.name] = (
                "clase" if isinstance(nodo, ast.ClassDef) else "función",
                firma + "\n" + ast.unparse(ast.Module(body=cuerpo,
                                                      type_ignores=[])))
        elif isinstance(nodo, (ast.Assign, ast.AnnAssign)):
            destino = (nodo.targets[0] if isinstance(nodo, ast.Assign)
                       else nodo.target)
            if isinstance(destino, ast.Name) and destino.id.isupper():
                salida[destino.id] = ("constante", ast.unparse(nodo.value))
    return salida


def equivalencia_refactor(ruta_motor, carpeta_src):
    """Compara cada definición de ``experimento.py`` con su copia en ``src/``.

    La comparación es sobre el árbol sintáctico, no sobre el texto: ignora
    comentarios, docstrings y formato, y normaliza los dos cambios de forma
    de la reorganización (ver ``_Normalizar``). Una fila «idéntica» significa
    que el código que se ejecuta es el mismo.
    """
    import difflib

    motor = _definiciones(ruta_motor)
    copias = {}
    for ruta in sorted(Path(carpeta_src).glob("*.py")):
        for nombre, (tipo, codigo) in _definiciones(ruta).items():
            copias[nombre] = (f"src/{ruta.name}", codigo)

    filas = []
    for nombre, (tipo, codigo) in motor.items():
        nuevo = RENOMBRADAS.get(nombre, nombre)
        if nuevo not in copias:
            filas.append({"definición": nombre, "tipo": tipo, "módulo": "—",
                          "resultado": "No trasladada", "diferencia": ""})
            continue
        modulo, codigo_src = copias[nuevo]
        if codigo_src == codigo:
            resultado, diferencia = "Idéntica", ""
        else:
            resultado = "Difiere"
            diferencia = " | ".join(
                linea.strip() for linea in difflib.unified_diff(
                    codigo.splitlines(), codigo_src.splitlines(),
                    lineterm="", n=0)
                if linea[:1] in "+-" and not linea.startswith(("+++", "---")))
        filas.append({"definición": nombre if nuevo == nombre
                      else f"{nombre} → {nuevo}",
                      "tipo": tipo, "módulo": modulo, "resultado": resultado,
                      "diferencia": diferencia})
    return pd.DataFrame(filas)


def resumen_pytest(*rutas_xml):
    """Resume uno o más reportes JUnit de pytest por archivo de pruebas.

    Devuelve una tabla por archivo y un diccionario con los totales y la
    fecha de ejecución de cada reporte. Las rutas que no existen se omiten.
    """
    import xml.etree.ElementTree as ET

    filas, totales = [], {"pruebas": 0, "fallidas": 0, "errores": 0,
                          "omitidas": 0, "segundos": 0.0, "ejecuciones": []}
    for ruta in rutas_xml:
        ruta = Path(ruta)
        if not ruta.exists():
            continue
        raiz = ET.parse(ruta).getroot()
        for suite in raiz.iter("testsuite"):
            totales["ejecuciones"].append(
                (ruta.name, suite.get("timestamp", "")[:16].replace("T", " ")))
            for caso in suite.iter("testcase"):
                estado = "aprobada"
                if caso.find("failure") is not None:
                    estado = "fallida"
                elif caso.find("error") is not None:
                    estado = "error"
                elif caso.find("skipped") is not None:
                    estado = "omitida"
                filas.append({"archivo": caso.get("classname", "").split(".")[-1]
                              + ".py",
                              "reporte": ruta.name, "estado": estado,
                              "segundos": float(caso.get("time", 0))})
    if not filas:
        return pd.DataFrame(), totales
    casos = pd.DataFrame(filas)
    totales.update(
        pruebas=len(casos),
        fallidas=int((casos["estado"] == "fallida").sum()),
        errores=int((casos["estado"] == "error").sum()),
        omitidas=int((casos["estado"] == "omitida").sum()),
        segundos=float(casos["segundos"].sum()))
    tabla = (casos.groupby("archivo")
             .agg(pruebas=("estado", "size"),
                  aprobadas=("estado", lambda s: int((s == "aprobada").sum())),
                  segundos=("segundos", "sum"))
             .reset_index())
    tabla["segundos"] = tabla["segundos"].round(1)
    return tabla, totales
