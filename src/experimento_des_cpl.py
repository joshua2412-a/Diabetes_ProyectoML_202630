"""
Protocolo experimental de DES-CPL (capítulo 13).

Evalúa el modelo original con las mismas particiones externas del
Entregable 2, para que cada fold sea una comparación emparejada con los
siete modelos base. Para cada fold externo hace dos cosas:

1. **Predicciones de los modelos base** (``ejecutar_fold``, la parte
   costosa). Cada modelo se construye con su mejor combinación del
   benchmark y con los hiperparámetros que eligió su propio bucle interno
   en ESE fold, de modo que DES-CPL no recibe información que los modelos
   base no tuvieron. Se calculan:

   * las predicciones fuera de pliegue sobre el entrenamiento externo, con
     tres pliegues agrupados por paciente (el conjunto de selección);
   * el modelo final ajustado con todo el entrenamiento externo, que
     reproduce el ajuste del Entregable 2 y predice el fold de evaluación;
   * el promedio de los tres modelos de pliegue sobre el fold de
     evaluación, que da el perfil de cada consulta.

   Se guarda en disco después de cada modelo: si se interrumpe, la
   siguiente ejecución continúa con el modelo que faltaba.

2. **Evaluación** (``evaluar_fold``, segundos). A partir de esas
   predicciones compara, en el fold de evaluación, los modelos base, dos
   ensambles estáticos (promedio simple y apilamiento), un ensamble con
   pesos globales y DES-CPL con sus variantes de ablación. Los
   hiperparámetros de cada método (``k`` y ``tau``) se eligen solo con el
   conjunto de selección, maximizando su AUC-PR; el fold de evaluación se
   usa una sola vez.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from src import configuracion as cfg
from src.balanceo import construir_pipeline
from src.des_cpl import (RegionCompetencia, combinar, competencia_global,
                         competencias_locales, perfil, pesos_softmax, podar)

#: Orden fijo de los modelos base (columnas de los archivos).
MODELOS = ("logistica", "bayes", "knn", "arbol", "random_forest", "xgboost",
           "svm")

#: Espacio de búsqueda de DES-CPL (valores candidatos de cada
#: hiperparámetro). Cada configuración se evalúa sobre las predicciones fuera
#: de pliegue, sin reentrenar ningún modelo.
#: Incluye los extremos de cada parámetro: temperatura 0 (selección dura) e
#: infinito (promedio), encogimiento 0 (solo local) y 1 (solo global),
#: poda a 4 modelos o ninguna (los 7).
K_REJILLA = (5, 10, 20, 30, 50, 100)
TAU_REJILLA = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, np.inf)
LAMBDA_REJILLA = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
PODA_REJILLA = (4, None)

#: Evaluaciones de hiperparámetros por búsqueda: el presupuesto validado en
#: el capítulo 12, el mismo que reciben los benchmarks con los que se
#: compara (Random Search a 60). La búsqueda es bayesiana (Optuna, TPE),
#: uno de los optimizadores del Entregable 2.
PRESUPUESTO = 60

#: Variantes. La primera es el modelo propuesto; la segunda, la versión
#: anterior al piloto; las demás cambian un solo componente (ablación).
VARIANTES = {
    "DES-CPL": {},
    "DES-CPL inicial (sin poda ni encogimiento)": {"lambdas": (0.0,), "podas": (None,)},
    "Ablación: sin encogimiento (λ = 0)": {"lambdas": (0.0,)},
    "Ablación: sin poda (7 modelos)": {"podas": (None,)},
    "Ablación: Brier balanceado": {"ponderacion": "balanceada"},
    "Ablación: competencia por exactitud": {"competencia": "exactitud"},
    "Ablación: región sin estratificar": {"estratificado": False},
    "Ablación: sin excluir al paciente": {"excluir_paciente": False},
    "Ablación: selección dura (τ = 0)": {"taus": (0.0,)},
}


# ---------------------------------------------------------------------------
# Combinaciones y particiones
# ---------------------------------------------------------------------------

def mejores_combinaciones(maestra):
    """Mejor combinación de cada modelo (regla del capítulo 11).

    Devuelve una tabla indexada por modelo con balanceo, optimizador, AUC-PR
    por fold del Entregable 2 y los hiperparámetros elegidos en cada fold.
    """
    completadas = maestra.query("estado == 'completada'")
    mejores = completadas.loc[completadas.groupby("modelo")["auc_pr_media"].idxmax()]
    tabla = mejores.set_index("modelo")[["balanceo", "optimizador",
                                         "auc_pr_media", "auc_pr_por_fold",
                                         "hiperparametros_por_fold"]].copy()
    tabla["auc_pr_por_fold"] = tabla["auc_pr_por_fold"].map(json.loads)
    tabla["hiperparametros_por_fold"] = tabla["hiperparametros_por_fold"].map(json.loads)
    return tabla.loc[[m for m in MODELOS if m in tabla.index]]


def con_presupuesto(mejores, ruta_evaluaciones, ruta_reajustes,
                    presupuesto=PRESUPUESTO):
    """Hiperparámetros de cada modelo y fold con el presupuesto del capítulo 12.

    El estudio de convergencia guardó, para la mejor combinación de cada
    modelo, las 60 configuraciones de Random Search de cada fold externo y
    el reajuste de cada una. Con ``presupuesto`` = 60 se toma la de mejor
    puntaje interno entre las 60, que es lo que habría elegido una búsqueda
    con ese presupuesto, y su AUC-PR externa. Así DES-CPL y los benchmarks
    con los que se compara reciben el mismo presupuesto de búsqueda.

    Los folds de esos archivos se numeran desde 1.
    """
    evaluaciones = pd.read_csv(ruta_evaluaciones)
    reajustes = pd.read_csv(ruta_reajustes).drop_duplicates(["modelo", "balanceo", "fold", "indice"])
    tabla = mejores.copy()
    for modelo in tabla.index:
        balanceo = tabla.loc[modelo, "balanceo"]
        parametros, auc = [], []
        for fold in range(1, len(tabla.loc[modelo, "hiperparametros_por_fold"]) + 1):
            ev = (evaluaciones.query("modelo == @modelo and balanceo == @balanceo and fold == @fold")
                  .drop_duplicates("indice").set_index("indice").sort_index())
            ev = ev[ev.index < presupuesto]
            if len(ev) < presupuesto:
                raise ValueError(f"{modelo}, fold {fold}: solo {len(ev)} de {presupuesto} evaluaciones")
            elegido = int(ev["puntaje_interno"].idxmax())
            fila = reajustes.query("modelo == @modelo and balanceo == @balanceo "
                                   "and fold == @fold and indice == @elegido")
            if fila.empty:
                raise ValueError(f"{modelo}, fold {fold}: falta el reajuste de la configuración {elegido}")
            parametros.append(json.loads(fila["parametros"].iloc[0]))
            auc.append(float(fila["auc_pr"].iloc[0]))
        tabla.at[modelo, "hiperparametros_por_fold"] = parametros
        tabla.at[modelo, "auc_pr_por_fold"] = auc
        tabla.at[modelo, "auc_pr_media"] = float(np.mean(auc))
    tabla["presupuesto"] = presupuesto
    return tabla


def particiones_externas(X, y, grupos, folds=5, semilla=cfg.SEMILLA):
    """Las particiones externas del Entregable 2 (mismo generador y semilla)."""
    externo = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                   random_state=semilla)
    return list(externo.split(X, y, groups=grupos))


def _rutas(carpeta, fold):
    carpeta = Path(carpeta)
    return (carpeta / f"seleccion_fold{fold}.csv",
            carpeta / f"evaluacion_fold{fold}.csv",
            carpeta / "tiempos.csv")


# ---------------------------------------------------------------------------
# Paso 1: predicciones de los modelos base
# ---------------------------------------------------------------------------

def ejecutar_fold(fold, particion, X, y, grupos, roles, mejores, carpeta,
                  razon_desbalance=None, n_pliegues=3, semilla=cfg.SEMILLA,
                  modelos=None, verbose=True):
    """Predicciones fuera de pliegue y del fold de evaluación de un fold externo.

    Escribe dos archivos en ``carpeta``:

    * ``seleccion_fold<i>.csv``: una fila por encuentro del entrenamiento
      externo, con su posición en ``X``, la etiqueta, el paciente y una
      columna ``oof_<modelo>`` por modelo.
    * ``evaluacion_fold<i>.csv``: una fila por encuentro del fold de
      evaluación, con ``final_<modelo>`` (modelo ajustado con todo el
      entrenamiento externo) y ``pliegue_<modelo>`` (promedio de los modelos
      de pliegue).

    Un modelo cuyas columnas ya están en disco se omite.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_sel, ruta_eval, ruta_tiempos = _rutas(carpeta, fold)
    modelos = list(modelos or mejores.index)
    entrenamiento, evaluacion = particion
    y = pd.Series(np.asarray(y), index=X.index)
    grupos = pd.Series(np.asarray(grupos), index=X.index)

    if ruta_sel.exists():
        seleccion = pd.read_csv(ruta_sel)
        evaluado = pd.read_csv(ruta_eval)
    else:
        seleccion = pd.DataFrame({"fila": entrenamiento,
                                  "y": y.iloc[entrenamiento].to_numpy(),
                                  "grupo": grupos.iloc[entrenamiento].to_numpy()})
        evaluado = pd.DataFrame({"fila": evaluacion,
                                 "y": y.iloc[evaluacion].to_numpy(),
                                 "grupo": grupos.iloc[evaluacion].to_numpy()})
    if not np.array_equal(seleccion["fila"], entrenamiento):
        raise ValueError(f"{ruta_sel} no corresponde a la partición del fold {fold}")

    X_tr, y_tr, g_tr = (X.iloc[entrenamiento], y.iloc[entrenamiento],
                        grupos.iloc[entrenamiento])
    X_ev = X.iloc[evaluacion]
    interno = StratifiedGroupKFold(n_splits=n_pliegues, shuffle=True,
                                   random_state=semilla)
    pliegues = list(interno.split(X_tr, y_tr, groups=g_tr))

    for modelo in modelos:
        if f"oof_{modelo}" in seleccion and f"final_{modelo}" in evaluado:
            if verbose:
                print(f"[omitido] fold {fold}, {modelo}: ya está en disco")
            continue
        fila = mejores.loc[modelo]
        parametros = fila["hiperparametros_por_fold"][fold]
        base = construir_pipeline(modelo, fila["balanceo"], roles, semilla,
                                  razon_desbalance).set_params(**parametros)
        inicio = time.perf_counter()

        oof = np.full(len(entrenamiento), np.nan)
        de_pliegue = np.zeros(len(evaluacion))
        for ajuste, validacion in pliegues:
            m = clone(base).fit(X_tr.iloc[ajuste], y_tr.iloc[ajuste])
            oof[validacion] = m.predict_proba(X_tr.iloc[validacion])[:, 1]
            de_pliegue += m.predict_proba(X_ev)[:, 1] / n_pliegues
        t_pliegues = time.perf_counter() - inicio

        inicio = time.perf_counter()
        final = clone(base).fit(X_tr, y_tr).predict_proba(X_ev)[:, 1]
        t_final = time.perf_counter() - inicio

        seleccion[f"oof_{modelo}"] = oof
        evaluado[f"final_{modelo}"] = final
        evaluado[f"pliegue_{modelo}"] = de_pliegue
        # Se escribe primero la evaluación: si se interrumpe entre los dos,
        # el modelo se repite (la condición de omisión exige ambos).
        evaluado.to_csv(ruta_eval, index=False, float_format="%.8g")
        seleccion.to_csv(ruta_sel, index=False, float_format="%.8g")
        pd.DataFrame([{"fold": fold, "modelo": modelo,
                       "t_pliegues_s": t_pliegues, "t_final_s": t_final}]).to_csv(
            ruta_tiempos, mode="a", header=not ruta_tiempos.exists(), index=False)
        if verbose:
            ap = average_precision_score(evaluado["y"], final)
            print(f"[listo] fold {fold}, {modelo}: AUC-PR {ap:.4f} "
                  f"(E2: {fila['auc_pr_por_fold'][fold]:.4f}) | pliegues "
                  f"{t_pliegues:.0f} s, final {t_final:.0f} s")
    return seleccion, evaluado


def progreso(carpeta, folds=range(5), modelos=MODELOS):
    """Qué modelos están calculados en cada fold."""
    filas = []
    for fold in folds:
        ruta_sel, ruta_eval, _ = _rutas(carpeta, fold)
        hechos = set()
        if ruta_sel.exists() and ruta_eval.exists():
            columnas = (set(pd.read_csv(ruta_sel, nrows=0).columns)
                        & {c.replace("final_", "oof_")
                           for c in pd.read_csv(ruta_eval, nrows=0).columns})
            hechos = {c.removeprefix("oof_") for c in columnas if c.startswith("oof_")}
        faltan = [m for m in modelos if m not in hechos]
        filas.append({"fold": fold, "modelos listos": len(modelos) - len(faltan),
                      "faltan": ", ".join(faltan) or "—",
                      "estado": "Completo" if not faltan else
                      ("Sin empezar" if len(faltan) == len(modelos) else "En curso")})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Paso 2: evaluación
# ---------------------------------------------------------------------------

def _metricas(y, p):
    return {"auc_pr": average_precision_score(y, p), "auc_roc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p)}


def _buscar(puntuar, espacio, n_evaluaciones=PRESUPUESTO, semilla=cfg.SEMILLA):
    """Búsqueda bayesiana (Optuna, TPE) sobre un espacio de valores candidatos.

    ``espacio`` asigna a cada hiperparámetro su lista de valores;
    ``puntuar(**config)`` devuelve el AUC-PR en el conjunto de selección.
    Devuelve ``(mejor puntaje, mejor configuración)``. Con la semilla fija,
    la búsqueda es reproducible.
    """
    optuna = cfg.requiere("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=semilla))

    def objetivo(prueba):
        config = {nombre: prueba.suggest_categorical(nombre, list(valores))
                  for nombre, valores in espacio.items()}
        return puntuar(**config)

    estudio.optimize(objetivo, n_trials=n_evaluaciones)
    return estudio.best_value, estudio.best_params


def evaluar_fold(seleccion, evaluado, modelos=MODELOS, ks=K_REJILLA,
                 taus=TAU_REJILLA, lambdas=LAMBDA_REJILLA, podas=PODA_REJILLA,
                 n_evaluaciones=PRESUPUESTO, semilla=cfg.SEMILLA):
    """Compara todos los métodos en un fold externo.

    Devuelve una fila por método con su familia, los hiperparámetros
    elegidos, el AUC-PR en el conjunto de selección con que se eligieron y
    las métricas en el fold de evaluación.
    """
    modelos = list(modelos)
    P_sel = seleccion[[f"oof_{m}" for m in modelos]].to_numpy()
    y_sel = seleccion["y"].to_numpy().astype(int)
    g_sel = seleccion["grupo"].to_numpy()
    P_ev = evaluado[[f"final_{m}" for m in modelos]].to_numpy()
    perfil_ev_todos = perfil(evaluado[[f"pliegue_{m}" for m in modelos]].to_numpy())
    y_ev = evaluado["y"].to_numpy().astype(int)
    perfil_sel = perfil(P_sel)
    filas = []

    def anotar(metodo, familia, p_ev, puntaje=np.nan, k=np.nan, tau=np.nan,
               lam=np.nan, modelos=""):
        filas.append({"método": metodo, "familia": familia, "k": k, "tau": tau,
                      "lambda": lam, "modelos_combinados": modelos,
                      "auc_pr_seleccion": puntaje, **_metricas(y_ev, p_ev)})

    for j, modelo in enumerate(modelos):
        anotar(modelo, "Modelo base", P_ev[:, j],
               average_precision_score(y_sel, P_sel[:, j]))

    anotar("Promedio simple", "Ensamble estático", P_ev.mean(axis=1),
           average_precision_score(y_sel, P_sel.mean(axis=1)))

    meta = LogisticRegression(max_iter=1000).fit(perfil_sel, y_sel)
    anotar("Apilamiento logístico", "Ensamble estático",
           meta.predict_proba(perfil(P_ev))[:, 1],
           average_precision_score(y_sel, meta.predict_proba(perfil_sel)[:, 1]))

    # Pesos globales: la misma softmax, con la competencia medida en todo el
    # conjunto de selección (Brier de cada modelo). Aísla lo que aporta que
    # la competencia sea local.
    brier_global = ((P_sel - y_sel[:, None]) ** 2).mean(axis=0, keepdims=True)
    def puntuar_global(tau):
        w = pesos_softmax(brier_global, tau)
        return average_precision_score(y_sel, combinar(w, P_sel))
    puntaje, config = _buscar(puntuar_global, {"tau": [float(t) for t in taus]},
                              n_evaluaciones, semilla)
    tau = config["tau"]
    w = pesos_softmax(brier_global, tau)
    anotar("Pesos globales (Brier)", "Ensamble estático",
           combinar(np.repeat(w, len(y_ev), axis=0), P_ev), puntaje, tau=tau)

    # DES-CPL. Para cada poda (que cambia el espacio de perfiles), el
    # vecindario de cada encuentro de selección (sin su paciente) y el de
    # cada consulta se calculan una sola vez con el k máximo y se recortan.
    k_max = max(ks)
    espacios = {}

    def espacio(poda):
        if poda not in espacios:
            activos = podar(P_sel, y_sel, poda)
            Ps, Pe = P_sel[:, activos], P_ev[:, activos]
            fs, fe = perfil(Ps), perfil_ev_todos[:, activos]
            region = RegionCompetencia(fs, y_sel, g_sel)
            # Ablación sin exclusión del paciente: solo se excluye el propio
            # encuentro.
            encuentros = np.arange(len(y_sel))
            region_enc = RegionCompetencia(fs, y_sel, encuentros)
            espacios[poda] = {
                "activos": activos, "Ps": Ps, "Pe": Pe, "region": region,
                "region_enc": region_enc,
                "vec_sel": region.vecindario(fs, k_max, g_sel),
                "vec_sel_enc": region_enc.vecindario(fs, k_max, encuentros),
                "vec_ev": region.vecindario(fe, k_max)}
        return espacios[poda]

    for nombre, opciones in VARIANTES.items():
        opciones = dict(opciones)
        taus_v = opciones.pop("taus", taus)
        lambdas_v = opciones.pop("lambdas", lambdas)
        podas_v = opciones.pop("podas", podas)
        excluir = opciones.pop("excluir_paciente", True)
        cache = {}

        def puntuar(poda, k, lam, tau):
            if (poda, k) not in cache:
                e = espacio(poda)
                region_b = e["region"] if excluir else e["region_enc"]
                vec_b = e["vec_sel"] if excluir else e["vec_sel_enc"]
                local = competencias_locales(region_b, vec_b, e["Ps"], k, **opciones)[0]
                glob = competencia_global(e["Ps"], y_sel, opciones.get("competencia", "brier"),
                                          opciones.get("ponderacion", "prevalencia"))
                cache[(poda, k)] = (local, glob)
            local, glob = cache[(poda, k)]
            c = (1 - lam) * local + lam * glob
            return average_precision_score(y_sel, combinar(pesos_softmax(c, tau),
                                                           espacio(poda)["Ps"]))

        puntaje, config = _buscar(
            puntuar, {"poda": list(podas_v), "k": [int(k) for k in ks],
                      "lam": [float(v) for v in lambdas_v],
                      "tau": [float(t) for t in taus_v]},
            n_evaluaciones, semilla)
        mejor = (puntaje, config["poda"], config["k"], config["lam"], config["tau"])
        puntaje, poda, k, lam, tau = mejor
        e = espacio(poda)
        c_ev = competencias_locales(e["region"], e["vec_ev"], e["Ps"], k,
                                    encogimiento=lam, **opciones)[0]
        familia = ("Modelo original" if nombre == "DES-CPL" else
                   "Versión inicial" if nombre.startswith("DES-CPL inicial") else "Ablación")
        anotar(nombre, familia, combinar(pesos_softmax(c_ev, tau), e["Pe"]),
               puntaje, k, tau, lam=lam,
               modelos=", ".join(modelos[i] for i in e["activos"]))
    return pd.DataFrame(filas)


def evaluar(carpeta, folds, ruta_resultados, modelos=MODELOS, **kwargs):
    """Evalúa los folds completos que aún no están en ``ruta_resultados``."""
    ruta_resultados = Path(ruta_resultados)
    hechos = (pd.read_csv(ruta_resultados) if ruta_resultados.exists()
              else pd.DataFrame(columns=["fold"]))
    listos = progreso(carpeta, folds, modelos).query("estado == 'Completo'")["fold"]
    for fold in listos:
        if fold in set(hechos["fold"]):
            continue
        ruta_sel, ruta_eval, _ = _rutas(carpeta, fold)
        opciones = {"semilla": cfg.SEMILLA + int(fold), **kwargs}
        tabla = evaluar_fold(pd.read_csv(ruta_sel), pd.read_csv(ruta_eval),
                             modelos, **opciones)
        tabla.insert(0, "fold", fold)
        hechos = pd.concat([hechos, tabla], ignore_index=True)
        hechos.to_csv(ruta_resultados, index=False)
    return hechos
