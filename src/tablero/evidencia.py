"""
Evidencia del modelo original para la página /modelo-original (pasos 3 a 6).

Todo se calcula con archivos que ya existen en results/ y artifacts/: nada se
reentrena ni se vuelve a evaluar. Las funciones puras reciben tablas y
devuelven tablas (se prueban con datos sintéticos); las que terminan en
``_csv`` o no reciben argumentos leen el CSV del repositorio y devuelven
``None`` si falta, para que la página muestre un aviso en esa sección y el
resto siga funcionando (Render solo ve lo que está en Git).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from src.estadistica import correccion_holm, delta_cliff, prueba_nadeau_bengio

from . import analitica as A
from . import config as C
from . import datos as D

#: Los siete modelos base y los dos del estado del arte: los nueve benchmarks
#: sobre los que el protocolo corrige con Holm.
BASE = ["logistica", "bayes", "knn", "arbol", "random_forest", "xgboost", "svm"]
SOTA = ["lightgbm", "catboost"]
BENCHMARKS = BASE + SOTA
ORIGINAL = C.MODELO_ORIGINAL
#: Razón entre observaciones de prueba y de entrenamiento de un fold de 5.
RAZON_NB = 0.25

#: Categorías de la diferencia DES-CPL − benchmark respecto de la franja de equivalencia.
MEJOR, PEOR, EQUIVALENTE, SIN_CONCLUSION = "mejor", "peor", "equivalente", "sin conclusión"


# =========================================================================== #
# Paso 3: el resultado
# =========================================================================== #
def matriz_por_fold(evaluacion: pd.DataFrame, sota: pd.DataFrame | None = None) -> pd.DataFrame:
    """AUC-PR por fold (filas) de cada método (columnas, con las claves del tablero)."""
    largo = A.por_fold_igual_presupuesto(evaluacion, sota)
    return largo.pivot(index="fold", columns="clave", values="auc_pr").sort_index()


def diferencias_con_intervalo(matriz: pd.DataFrame, equivalencia: float = C.EQUIVALENCIA_AUC_PR,
                              benchmarks: list[str] | None = None) -> pd.DataFrame:
    """Diferencia DES-CPL − cada benchmark, con su intervalo del 95 % y Holm.

    Varianza de Nadeau y Bengio (2003): ``(1/k + 1/4) s²`` con ``k`` folds
    (factor 1/5 + 1/4 con 5) y t con ``k - 1`` grados de libertad, que es la
    corrección que protege de que los folds comparten entrenamiento. El valor p
    es el unilateral de ``prueba_nadeau_bengio`` (H1: DES-CPL > benchmark) y
    Holm se aplica sobre todos los benchmarks de la tabla (nueve con los
    del protocolo).

    La categoría compara el intervalo con la franja ±``equivalencia``:
    «mejor» o «peor» si el intervalo excluye el cero; «equivalente» si cae
    entero dentro de la franja; «sin conclusión» si cruza el cero y se sale de
    la franja (con 5 folds, lo más frecuente).
    """
    benchmarks = [b for b in (benchmarks or BENCHMARKS) if b in matriz.columns]
    k = len(matriz)
    t_critico = float(stats.t.ppf(0.975, k - 1))
    filas = []
    for b in benchmarks:
        d = (matriz[ORIGINAL] - matriz[b]).to_numpy()
        media, sd = float(d.mean()), float(d.std(ddof=1))
        mitad = t_critico * np.sqrt((1 / k + RAZON_NB) * sd ** 2)
        prueba = prueba_nadeau_bengio(d, RAZON_NB)
        filas.append({"clave": b, "benchmark": C.nombre_modelo(b), "diferencia": media, "sd": sd,
                      "ic_bajo": media - mitad, "ic_alto": media + mitad, "t": prueba["t"], "p": prueba["p"],
                      "gana_folds": int((d > 0).sum()), "n_folds": k,
                      "cliff": float(delta_cliff(matriz[ORIGINAL], matriz[b]))})
    tabla = pd.DataFrame(filas)
    tabla["p_holm"] = correccion_holm(tabla["p"]) if len(tabla) else []

    def categoria(f):
        if f["ic_bajo"] > 0:
            return MEJOR
        if f["ic_alto"] < 0:
            return PEOR
        if f["ic_bajo"] >= -equivalencia and f["ic_alto"] <= equivalencia:
            return EQUIVALENTE
        return SIN_CONCLUSION

    tabla["categoria"] = tabla.apply(categoria, axis=1) if len(tabla) else []
    return tabla


def lectura_diferencias(tabla: pd.DataFrame, equivalencia: float = C.EQUIVALENCIA_AUC_PR) -> list[str]:
    """Frases del gráfico de diferencias, calculadas con la tabla."""
    from .formato import num

    mejor = tabla.sort_values("diferencia", ascending=False)
    peor_rival = tabla.loc[tabla["diferencia"].idxmin()]
    significativas = tabla[tabla["p_holm"] < 0.05]
    lineas = []
    if (tabla["clave"] == "xgboost").any():
        x = tabla[tabla["clave"] == "xgboost"].iloc[0]
        lineas.append(f"Frente a XGBoost, el mejor benchmark, la diferencia es {num(x['diferencia'], 4, signo=True)} "
                      f"(IC 95 %: {num(x['ic_bajo'], 4)} a {num(x['ic_alto'], 4)}; p = {num(x['p'], 2)}).")
    lineas.append(f"Con Holm sobre los {len(tabla)} benchmarks, DES-CPL es significativamente mejor que "
                  f"{len(significativas)} y no es significativamente peor que ninguno."
                  if (tabla["categoria"] != PEOR).all() else
                  f"Con Holm sobre los {len(tabla)} benchmarks, DES-CPL es significativamente mejor que "
                  f"{len(significativas)}; frente a {peor_rival['benchmark']} la diferencia es negativa.")
    sin = int((tabla["categoria"] == SIN_CONCLUSION).sum())
    equiv = int((tabla["categoria"] == EQUIVALENTE).sum())
    lineas.append(f"Con 5 folds, {sin} intervalos cruzan el cero y se salen de la franja de ±{num(equivalencia, 3)}: "
                  f"los datos no bastan para declarar equivalencia ni diferencia"
                  + (f" ({equiv} caen dentro de la franja)." if equiv else "."))
    return lineas


def legible(texto: str) -> str:
    """Coma decimal y nombres de modelo legibles en un texto de results/ o artifacts/.

    Esos archivos siguen la convención del libro (punto decimal y claves como
    «xgboost»); el tablero escribe con coma y con los nombres de ``C.MODELOS``.
    """
    import re

    from .formato import decimales_es

    texto = decimales_es(str(texto))
    return re.sub(r"\b(arbol|random_forest|xgboost|svm|logistica|knn|bayes|lightgbm|catboost)\b",
                  lambda m: C.nombre_modelo(m.group(1)), texto)


def pruebas_protocolo() -> pd.DataFrame | None:
    """Las tres pruebas confirmatorias del protocolo (Friedman con Nemenyi,
    Nadeau-Bengio y DeLong), tal como las dejó el capítulo 13."""
    tabla = D.leer_csv_opcional(C.RESULTADOS / "pruebas_hipotesis_modelo_original.csv")
    if tabla is not None:
        return tabla.drop(columns=[c for c in tabla.columns if c.startswith("Unnamed")])
    datos = D.archivo_original("pruebas_hipotesis.json")
    return pd.DataFrame(datos) if datos else None


def condiciones_superioridad(matriz: pd.DataFrame | None, pruebas: pd.DataFrame | None,
                             adicionales: pd.DataFrame | None) -> list[dict]:
    """Las cuatro condiciones de la regla de superioridad (protocolo, sección 5).

    Replica la celda del capítulo 17.2 con los mismos archivos. Cada condición
    es un dict con ``texto``, ``cumple`` (bool, o None si faltan los datos) y
    ``valor`` (la cifra que la decide). Solo hay superioridad si se cumplen las
    cuatro a la vez.
    """
    from .formato import num, p_valor

    condiciones = []
    # 1. Media mayor que la del mejor benchmark y que la de cada modelo base.
    if matriz is not None and ORIGINAL in matriz:
        medias = matriz.mean()
        rivales = [b for b in BENCHMARKS if b in medias.index]
        mejor = medias[rivales].idxmax()
        c1 = bool(medias[ORIGINAL] > medias[rivales].max())
        v1 = f"DES-CPL {num(medias[ORIGINAL], 4)} frente a {C.nombre_modelo(mejor)} {num(medias[mejor], 4)}"
    else:
        c1, v1, mejor = None, "sin datos", None
    condiciones.append({"texto": "AUC-PR media mayor que la del mejor benchmark y la de cada modelo base",
                        "cumple": c1, "valor": v1})
    # 2. Las tres pruebas rechazan H0 con p ajustado < 0,05.
    if pruebas is not None and "p ajustado (Holm)" in pruebas:
        p = pd.to_numeric(pruebas["p ajustado (Holm)"], errors="coerce")
        separa = "no separa" not in str(pruebas.iloc[0].get("conclusión", ""))
        c2 = bool((p < 0.05).all()) and separa
        rechazan = int((p.to_numpy() < 0.05).sum()) - (0 if separa else 1)
        detalle = "; ".join(f"{str(n).split(' (')[0]}: {p_valor(x)}" for n, x in zip(pruebas["prueba"], p))
        v2 = f"{rechazan} de {len(p)} rechazan frente al mejor ({detalle}"
        v2 += ")" if separa else "; Nemenyi no separa a DES-CPL del mejor)"
    else:
        c2, v2 = None, "sin datos"
    condiciones.append({"texto": "Las tres pruebas rechazan H0 con p ajustado < 0,05", "cumple": c2, "valor": v2})
    # 3. |δ de Cliff| ≥ 0,147 a favor de DES-CPL frente al mejor benchmark.
    if matriz is not None and mejor is not None:
        cliff = float(delta_cliff(matriz[ORIGINAL], matriz[mejor]))
        c3, v3 = bool(cliff >= 0.147), f"δ de Cliff = {num(cliff, 2)} frente a {C.nombre_modelo(mejor)}"
    else:
        c3, v3 = None, "sin datos"
    condiciones.append({"texto": "δ de Cliff de al menos 0,147 a favor de DES-CPL", "cumple": c3, "valor": v3})
    # 4. Mejor en ≥ 2 de los 3 conjuntos adicionales y no peor con significancia en ninguno.
    if adicionales is not None:
        gana, peor = 0, 0
        conjuntos = 0
        for _, datos in adicionales.groupby("conjunto"):
            tabla = datos.pivot_table(index="fold", columns="método", values="auc_pr")
            rivales = [b for b in BENCHMARKS if b in tabla.columns]
            mejor_c = tabla[rivales].mean().idxmax()
            d = tabla["DES-CPL"] - tabla[mejor_c]
            conjuntos += 1
            gana += int(d.mean() > 0)
            peor += int(prueba_nadeau_bengio(-d)["p"] < 0.05)
        c4 = bool(gana >= 2 and peor == 0)
        v4 = f"Mejor en {gana} de {conjuntos}; significativamente peor en {peor}"
    else:
        c4, v4 = None, "sin datos"
    condiciones.append({"texto": "Mejor en al menos 2 de 3 conjuntos adicionales y no significativamente peor en ninguno",
                        "cumple": c4, "valor": v4})
    return condiciones


# =========================================================================== #
# Paso 4: por qué empata
# =========================================================================== #
def lambda_por_fold(evaluacion: pd.DataFrame) -> pd.Series:
    """λ que eligió la búsqueda en cada fold (columna ``lambda`` de DES-CPL)."""
    original = evaluacion[evaluacion["familia"] == "Modelo original"].sort_values("fold")
    return original.set_index("fold")["lambda"].astype(float)


def ablacion(evaluacion: pd.DataFrame) -> pd.DataFrame:
    """Diferencia (variante − DES-CPL completo) por fold, en media ± desviación.

    Usa las familias «Ablación» y «Versión inicial» del CSV. Negativo = quitar
    ese componente empeora; positivo = lo mejora.
    """
    completo = evaluacion[evaluacion["familia"] == "Modelo original"].set_index("fold")["auc_pr"]
    filas = []
    for metodo, datos in evaluacion[evaluacion["familia"].isin(["Ablación", "Versión inicial"])].groupby("método"):
        d = datos.set_index("fold")["auc_pr"] - completo
        filas.append({"variante": metodo, "familia": datos["familia"].iloc[0], "diferencia": float(d.mean()),
                      "sd": float(d.std(ddof=1)), "n_folds": int(d.notna().sum())})
    return pd.DataFrame(filas).sort_values("diferencia").reset_index(drop=True)


def diagnostico_r2() -> dict | None:
    """Mayor R² fuera de pliegue al predecir qué modelo acierta (results/des_cpl_diagnostico_errores.csv)."""
    tabla = D.leer_csv_opcional(C.RESULTADOS / "des_cpl_diagnostico_errores.csv")
    if tabla is None:
        return None
    r2 = tabla[tabla["análisis"].str.startswith("R² de pérdida")]
    if r2.empty:
        return None
    return {"maximo": float(r2["valor_seleccion"].max()), "n_pares": int(len(r2)),
            "espacios": sorted(r2["espacio"].unique())}


# =========================================================================== #
# Paso 5: generalización y estabilidad
# =========================================================================== #
def rangos_en_conjuntos(evaluacion: pd.DataFrame, sota: pd.DataFrame | None,
                        adicionales: pd.DataFrame) -> pd.DataFrame:
    """Rango medio de cada método (1 = mejor AUC-PR) en el problema principal
    más los conjuntos adicionales, y solo en los adicionales.

    Columnas: ``clave``, ``rango_4`` (los 4 conjuntos) y ``rango_3`` (los 3
    adicionales). El AUC-PR de cada conjunto es la media de sus 5 folds.
    """
    principal = A.por_fold_igual_presupuesto(evaluacion, sota).groupby("clave")["auc_pr"].mean()
    tablas = {"diabetes": principal}
    extra = adicionales.copy()
    extra["clave"] = extra["método"].replace({"DES-CPL": ORIGINAL})
    extra = extra[extra["familia"].isin(["Modelo base", "Ensamble estático", "Modelo original", "Estado del arte"])]
    for conjunto, datos in extra.groupby("conjunto"):
        tablas[conjunto] = datos.groupby("clave")["auc_pr"].mean()
    medias = pd.DataFrame(tablas).dropna()
    resultado = pd.DataFrame({
        "rango_4": medias.rank(ascending=False).mean(axis=1),
        # Los 3 adicionales solos: se reordenan sin el problema principal.
        "rango_3": medias.drop(columns="diabetes").rank(ascending=False).mean(axis=1)})
    resultado["n_conjuntos"] = medias.shape[1]
    resultado = resultado.reset_index(names="clave")
    resultado["metodo"] = resultado["clave"].map(C.nombre_modelo)
    return resultado.sort_values("rango_4").reset_index(drop=True)


def robustez_perdidas(tabla: pd.DataFrame) -> pd.DataFrame:
    """Pérdida de AUC-PR respecto del escenario original (original − escenario).

    ``tabla`` es results/robustez_auc_pr.csv. Positivo = pierde. Devuelve una
    fila por método y escenario, en formato largo, y la razón de cada método con
    DES-CPL (cuántas veces pierde lo que pierde DES-CPL).
    """
    escenarios = [c for c in tabla.columns if c not in ("método", "original")]
    largo = tabla.melt(id_vars=["método", "original"], value_vars=escenarios,
                       var_name="escenario", value_name="auc_pr")
    largo["perdida"] = largo["original"] - largo["auc_pr"]
    referencia = largo[largo["método"] == "DES-CPL"].set_index("escenario")["perdida"]
    largo["razon_con_des_cpl"] = largo["perdida"] / largo["escenario"].map(referencia)
    largo["clave"] = largo["método"].replace({"DES-CPL": ORIGINAL}).map(C.nombre_modelo)
    return largo


ESCENARIOS = {"ruido_05": "5 % de etiquetas cambiadas", "ruido_10": "10 % de etiquetas cambiadas",
              "prevalencia_05": "Prevalencia reducida al 5 %", "tamano_50": "50 % de los pacientes",
              "tamano_25": "25 % de los pacientes"}


def sintesis_comprobaciones() -> dict:
    """Lee las síntesis de las verificaciones (V) y exploraciones (E), si existen.

    Devuelve ``{"verificaciones": {"total", "mantienen", "no_mantienen"},
    "exploraciones": {"E1": {...}}}``, o ``None`` en la clave que falte.
    """
    salida = {"verificaciones": None, "exploraciones": None}
    v = D.leer_csv_opcional(C.RESULTADOS / "verificaciones" / "sintesis.csv")
    if v is not None and {"verificación", "se mantiene la conclusión"} <= set(v.columns):
        mantiene = v["se mantiene la conclusión"].astype(str).str.strip().str.lower().isin(["sí", "si", "true"])
        salida["verificaciones"] = {"total": int(len(v)), "mantienen": int(mantiene.sum()),
                                    "no_mantienen": v.loc[~mantiene, "verificación"].tolist()}
    e = D.leer_csv_opcional(C.RESULTADOS / "exploracion" / "sintesis.csv")
    if e is not None and len(e):
        primera = e.iloc[0]
        salida["exploraciones"] = {"nombre": str(primera.iloc[0]), "auc_pr": float(primera["AUC-PR"]),
                                   "diferencia": float(primera["dif. media"]), "p": float(primera["p (unilateral)"]),
                                   "n": int(len(e))}
    return salida


def semillas_resumen() -> pd.DataFrame | None:
    tabla = D.leer_csv_opcional(C.RESULTADOS / "semillas_resumen.csv")
    if tabla is None:
        return None
    tabla = tabla.copy()
    tabla["metodo"] = tabla["método"].replace({"DES-CPL": ORIGINAL}).map(C.nombre_modelo)
    return tabla
