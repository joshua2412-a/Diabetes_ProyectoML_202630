"""
Cálculos de las vistas clínicas y de modelos del dashboard.

Este módulo solo calcula tablas a partir de los datos y los artefactos: no
dibuja (eso es graficos_clinicos.py) ni toca la interfaz (pages/). Todo se
calcula con los mismos datos que el resto del tablero: el entrenamiento
para el EDA y las predicciones de prueba guardadas para los modelos. Nada
reentrena ni vuelve a evaluar un modelo, salvo las predicciones de un caso
concreto que pide el usuario.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from . import datos as D
from .formato import num

Y = "readmitted_30"


# =========================================================================== #
# Utilidades
# =========================================================================== #
def wilson(k, n, z: float = 1.96):
    """Intervalo de Wilson al 95 % para una proporción.

    Se prefiere al intervalo de Wald porque no se sale de [0, 1] ni colapsa
    en los grupos pequeños o con tasas cercanas a cero, que en este conjunto
    son frecuentes (subgrupos de la cola de las distribuciones).
    """
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        centro = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
        radio = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    # El redondeo puede dejar -1e-17 en el extremo: se recorta a [0, 1].
    return np.clip(centro - radio, 0, 1), np.clip(centro + radio, 0, 1)


def resumen_tasa(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Encuentros, readmitidos, tasa (%) e IC de Wilson por nivel."""
    tabla = df.groupby(columna, observed=True)[Y].agg(n="size", k="sum").reset_index()
    tabla["tasa"] = 100 * tabla["k"] / tabla["n"]
    bajo, alto = wilson(tabla["k"], tabla["n"])
    tabla["ic_bajo"], tabla["ic_alto"] = 100 * bajo, 100 * alto
    return tabla


def personas_de_100(tasa: float) -> int:
    """Cuántas de cada 100 personas, redondeando sin perder los extremos.

    Una tasa de 0,4 % no debe mostrarse como «0 de cada 100» si hay casos:
    se muestra como 1; y una de 99,6 %, como 99.
    """
    n = int(round(tasa))
    if 0 < tasa < 1:
        return 1
    if 99 < tasa < 100:
        return 99
    return min(100, max(0, n))


# =========================================================================== #
# EDA clínico
# =========================================================================== #
#: Etapas del recorrido del paciente (Sankey).
ESTANCIA = [(1, 3, "Estancia corta (1–3 días)"), (4, 7, "Estancia media (4–7 días)"),
            (8, 14, "Estancia larga (8–14 días)")]


def _estancia(dias: pd.Series) -> pd.Series:
    etiquetas = pd.Series(ESTANCIA[-1][2], index=dias.index)
    for desde, hasta, nombre in ESTANCIA:
        etiquetas[dias.between(desde, hasta)] = nombre
    return etiquetas


def recorrido(df: pd.DataFrame, minimo: int = 30) -> dict:
    """Flujos admisión → estancia → destino al alta → desenlace.

    Los niveles con menos de ``minimo`` encuentros en una etapa se agrupan en
    «Otro»: una cinta de tres pacientes no se ve y su tasa no es estable.
    """
    etapas = pd.DataFrame({
        "Admisión": df["admission_type_id"].astype(str).map(lambda v: C.nivel("admission_type_id", v)),
        "Estancia": _estancia(df["time_in_hospital"]),
        "Alta": df["discharge_disposition_id"].astype(str).map(
            lambda v: C.nivel("discharge_disposition_id", v)),
        "Desenlace": df[Y].map(C.CLASES_CORTAS),
        Y: df[Y].to_numpy(),
    })
    for etapa in ["Admisión", "Alta"]:
        conteo = etapas[etapa].value_counts()
        raros = conteo[conteo < minimo].index
        etapas.loc[etapas[etapa].isin(raros), etapa] = "Otro"
    columnas = ["Admisión", "Estancia", "Alta", "Desenlace"]
    nodos, indice = [], {}
    for etapa in columnas:
        orden = (etapas[etapa].value_counts().index.tolist() if etapa != "Estancia"
                 else [e[2] for e in ESTANCIA if e[2] in set(etapas[etapa])])
        if etapa == "Desenlace":
            orden = [C.CLASES_CORTAS[0], C.CLASES_CORTAS[1]]
        for nivel in orden:
            indice[(etapa, nivel)] = len(nodos)
            sub = etapas[etapas[etapa] == nivel]
            nodos.append({"etapa": etapa, "nivel": nivel, "n": len(sub),
                          "tasa": 100 * sub[Y].mean() if len(sub) else np.nan})
    enlaces = []
    for origen, destino in zip(columnas[:-1], columnas[1:]):
        flujo = etapas.groupby([origen, destino], observed=True)[Y].agg(n="size", k="sum").reset_index()
        for _, f in flujo.iterrows():
            enlaces.append({"origen": indice[(origen, f[origen])], "destino": indice[(destino, f[destino])],
                            "n": int(f["n"]), "tasa": 100 * f["k"] / f["n"],
                            "final": destino == "Desenlace",
                            "clase": int(f[destino] == C.CLASES_CORTAS[1]) if destino == "Desenlace" else None})
    return {"nodos": pd.DataFrame(nodos), "enlaces": pd.DataFrame(enlaces)}


def diagnosticos(df: pd.DataFrame, minimo: int = 60) -> pd.DataFrame:
    """Jerarquía diagnóstico principal → secundario con tamaño y tasa.

    Los secundarios con menos de ``minimo`` encuentros dentro de su
    principal se agrupan en «Otros secundarios», para que el sunburst no se
    llene de porciones invisibles con tasas inestables.
    """
    filas = []
    for principal, sub in df.groupby("diag_1_grupo", observed=True):
        if sub.empty:
            continue
        filas.append({"id": str(principal), "padre": "", "etiqueta": str(principal),
                      "n": len(sub), "tasa": 100 * sub[Y].mean()})
        segundo = sub.groupby("diag_2_grupo", observed=True)[Y].agg(n="size", k="sum")
        grandes, chicos = segundo[segundo["n"] >= minimo], segundo[segundo["n"] < minimo]
        for nombre_s, f in grandes.iterrows():
            filas.append({"id": f"{principal}/{nombre_s}", "padre": str(principal),
                          "etiqueta": str(nombre_s), "n": int(f["n"]), "tasa": 100 * f["k"] / f["n"]})
        if chicos["n"].sum() > 0:
            filas.append({"id": f"{principal}/__otros", "padre": str(principal),
                          "etiqueta": "Otros secundarios", "n": int(chicos["n"].sum()),
                          "tasa": 100 * chicos["k"].sum() / chicos["n"].sum()})
    return pd.DataFrame(filas)


def edad_legible(codigo) -> str:
    """Decenio de edad en palabras: el código 2 («[20-30)») pasa a «20–29»."""
    texto = D.etiquetas_edad().get(int(codigo), str(codigo)).strip("[)")
    try:
        desde, hasta = (int(v) for v in texto.split("-"))
    except ValueError:
        return texto
    return f"{desde}–{hasta - 1}"


def piramide(df: pd.DataFrame) -> pd.DataFrame:
    """Encuentros y tasa por decenio de edad y sexo."""
    tabla = df.groupby(["age_ord", "gender"], observed=True)[Y].agg(n="size", k="sum").reset_index()
    tabla["tasa"] = 100 * tabla["k"] / tabla["n"]
    tabla["edad"] = tabla["age_ord"].map(edad_legible)
    tabla["sexo"] = tabla["gender"].astype(str).map(lambda v: C.nivel("gender", v))
    return tabla


#: Variables de «dosis-respuesta»: conteos donde tiene sentido preguntar
#: «¿y con un valor más?». El tope agrupa la cola larga (≥ tope).
DOSIS = {"number_inpatient": 6, "number_emergency": 4, "number_outpatient": 5,
         "number_diagnoses": 9, "time_in_hospital": 12, "num_medications": 40,
         "num_procedures": 6, "n_farmacos_activos": 4}


def dosis_respuesta(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Tasa de readmisión (con IC) por valor de un conteo, con la cola agrupada.

    ``num_medications`` tiene demasiados valores para leerlos uno a uno: se
    agrupa de 5 en 5 hasta el tope.
    """
    tope = DOSIS.get(variable, int(df[variable].quantile(0.95)))
    valores = df[variable].clip(upper=tope)
    if variable == "num_medications":
        valores = (valores // 5) * 5
    datos = pd.DataFrame({"valor": valores, Y: df[Y]})
    tabla = resumen_tasa(datos, "valor").sort_values("valor")
    paso = 5 if variable == "num_medications" else 1
    tabla["etiqueta"] = [
        (f"≥ {int(v)}" if v >= tope else (f"{int(v)}–{int(v) + paso - 1}" if paso > 1 else str(int(v))))
        for v in tabla["valor"]]
    return tabla


#: Estados comunes a todos los fármacos. El ETL agrupó «sube» y «baja» en
#: «Cambio» para los fármacos poco frecuentes, pero no para la insulina: aquí
#: se agrupan también los de la insulina, para que todas las filas tengan las
#: mismas tres columnas (el detalle sube/baja va en la lectura).
ESTADOS_COMUNES = {"No": "No", "Steady": "Steady", "Up": "Cambio", "Down": "Cambio", "Cambio": "Cambio"}


def farmacos(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de readmisión por fármaco y estado de prescripción (formato largo).

    ``estado`` es el estado común (No, Steady, Cambio); ``estado_original``
    conserva Up y Down de la insulina para la lectura.
    """
    filas = []
    for farmaco in D.roles()["farmacos"]:
        if farmaco not in df:
            continue
        original = df[farmaco].astype(str)
        comun = original.map(ESTADOS_COMUNES).fillna(original)
        prescritos = int((comun != "No").sum())
        for columna, valores in [("estado", comun), ("estado_original", original)]:
            resumen = df[Y].groupby(valores.to_numpy()).agg(n="size", k="sum")
            for estado, f in resumen.iterrows():
                if columna == "estado_original" and estado not in ("Up", "Down"):
                    continue
                filas.append({"farmaco": farmaco, "nombre": C.nombre(farmaco), "tipo": columna,
                              "estado": str(estado),
                              "estado_legible": C.ESTADOS_FARMACO.get(str(estado), str(estado)),
                              "n": int(f["n"]), "k": int(f["k"]), "tasa": 100 * f["k"] / f["n"],
                              "prescritos": prescritos})
    return pd.DataFrame(filas)


def hba1c(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de readmisión por resultado de HbA1c y cambio de medicación."""
    datos = df[["A1Cresult", "change", Y]].copy()
    datos["A1Cresult"] = datos["A1Cresult"].astype(str)
    datos["change"] = datos["change"].astype(str)
    tabla = datos.groupby(["A1Cresult", "change"], observed=True)[Y].agg(n="size", k="sum").reset_index()
    tabla["tasa"] = 100 * tabla["k"] / tabla["n"]
    bajo, alto = wilson(tabla["k"], tabla["n"])
    tabla["ic_bajo"], tabla["ic_alto"] = 100 * bajo, 100 * alto
    orden = {"No medida": 0, "Norm": 1, ">7": 2, ">8": 3}
    tabla["orden"] = tabla["A1Cresult"].map(orden).fillna(9)
    return tabla.sort_values(["orden", "change"])


#: Indicadores del comparador de perfiles: (etiqueta, función sobre el
#: DataFrame que devuelve una máscara booleana). Todos en porcentaje, para
#: que las barras enfrentadas compartan escala.
INDICADORES = [
    ("Readmitidos en < 30 días", lambda d: d[Y] == 1),
    ("Hospitalización el año previo", lambda d: d["number_inpatient"] > 0),
    ("Urgencias el año previo", lambda d: d["number_emergency"] > 0),
    ("Ingreso por emergencia", lambda d: d["admission_type_id"].astype(str).str.strip() == "Emergency"),
    ("Estancia de 7 días o más", lambda d: d["time_in_hospital"] >= 7),
    ("20 medicamentos o más", lambda d: d["num_medications"] >= 20),
    ("Con insulina", lambda d: d["insulin"].astype(str) != "No"),
    ("Cambio de medicación", lambda d: d["change"].astype(str) == "Ch"),
    ("HbA1c medida", lambda d: d["A1Cresult"].astype(str) != "No medida"),
    ("Alta a casa", lambda d: d["discharge_disposition_id"].astype(str).str.startswith("Discharged to home")),
]


def perfil(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(np.nan, index=[i[0] for i in INDICADORES])
    return pd.Series({etiqueta: 100 * float(f(df).mean()) for etiqueta, f in INDICADORES})


def filtrar_perfil(df: pd.DataFrame, edades=None, sexos=None, previas=None, diagnostico=None) -> pd.DataFrame:
    """Subgrupo del comparador, aplicado sobre lo que dejan los filtros globales."""
    mascara = np.ones(len(df), dtype=bool)
    if edades:
        mascara &= df["age_ord"].isin([int(e) for e in edades]).to_numpy()
    if sexos:
        mascara &= df["gender"].astype(str).isin(sexos).to_numpy()
    if previas:
        mascara &= df["number_inpatient"].between(previas[0], 99 if previas[1] >= 5 else previas[1]).to_numpy()
    if diagnostico:
        mascara &= df["diag_1_grupo"].astype(str).isin(diagnostico).to_numpy()
    return df[mascara]


def categorias_paralelas(df: pd.DataFrame, dimensiones: list[str], maximo: int = 6) -> pd.DataFrame:
    """Combinaciones de niveles con su conteo, para ``go.Parcats``.

    Se agrega antes de enviar al navegador: mandar los ~80.000 encuentros
    fila a fila pesaría varios MB por interacción. Cada dimensión conserva
    sus ``maximo - 1`` niveles más frecuentes y agrupa el resto en «Otros».
    """
    datos = pd.DataFrame(index=df.index)
    for d in dimensiones:
        if d == "age_ord":
            serie = df[d].map(edad_legible)
        elif d == "number_inpatient":
            serie = df[d].clip(upper=3).map(lambda v: "≥ 3" if v >= 3 else str(int(v)))
        elif d == "time_in_hospital":
            serie = _estancia(df[d]).str.replace(r" \(.*\)", "", regex=True)
        else:
            serie = df[d].astype(str).map(lambda v, d=d: C.nivel(d, v))
        frecuentes = serie.value_counts().index[: maximo - 1]
        datos[d] = serie.where(serie.isin(frecuentes), "Otros")
    datos[Y] = df[Y].to_numpy()
    return datos.groupby(dimensiones + [Y], observed=True).size().rename("n").reset_index()


# =========================================================================== #
# Modelos
# =========================================================================== #
def _yp(modelo: str):
    pred = D.predicciones_prueba()
    return pred["y_real"].to_numpy(), pred[modelo].to_numpy(dtype=float)


@lru_cache(maxsize=16)
def curva_capacidad(modelo: str) -> pd.DataFrame:
    """Detección y precisión al marcar al k % de mayor riesgo (k = 1..100).

    Es la curva de ganancia acumulada: «revisando al 20 % de mayor riesgo se
    captura el X % de las readmisiones». Se ordena por la probabilidad, así
    que no depende de su calibración.
    """
    y, p = _yp(modelo)
    orden = np.argsort(-p, kind="stable")
    acumulado = np.cumsum(y[orden])
    n, positivos = len(y), y.sum()
    filas = []
    for k in range(1, 101):
        marcados = max(1, int(round(n * k / 100)))
        aciertos = int(acumulado[marcados - 1])
        filas.append({"porcentaje": k, "marcados": marcados, "aciertos": aciertos,
                      "deteccion": 100 * aciertos / positivos, "precision": 100 * aciertos / marcados,
                      "umbral": float(p[orden][marcados - 1])})
    return pd.DataFrame(filas)


def capacidad(modelo: str, porcentaje: int) -> dict:
    """Matriz de confusión cuando el servicio sigue al ``porcentaje`` % de mayor riesgo."""
    y, p = _yp(modelo)
    fila = curva_capacidad(modelo).set_index("porcentaje").loc[int(porcentaje)]
    marcado = np.zeros(len(y), dtype=bool)
    marcado[np.argsort(-p, kind="stable")[: int(fila["marcados"])]] = True
    vp = int((marcado & (y == 1)).sum())
    fp = int((marcado & (y == 0)).sum())
    fn = int((~marcado & (y == 1)).sum())
    vn = int((~marcado & (y == 0)).sum())
    return {"vp": vp, "fp": fp, "fn": fn, "vn": vn, "n": len(y), "umbral": float(fila["umbral"]),
            "deteccion": 100 * vp / max(1, vp + fn), "precision": 100 * vp / max(1, vp + fp),
            "por_readmision": (vp + fp) / vp if vp else np.inf}


def a_cien(conteos: dict) -> dict:
    """Reparte 100 personas entre los grupos con el método del mayor resto.

    Así los cuatro grupos suman exactamente 100 y ningún grupo con casos
    queda en cero por redondeo (si hay sitio).
    """
    total = sum(conteos.values())
    cuotas = {k: 100 * v / total for k, v in conteos.items()}
    enteros = {k: int(np.floor(c)) for k, c in cuotas.items()}
    for k, v in conteos.items():
        if v > 0 and enteros[k] == 0:
            enteros[k] = 1
    sobrante = 100 - sum(enteros.values())
    restos = sorted(cuotas, key=lambda k: cuotas[k] - np.floor(cuotas[k]), reverse=True)
    i = 0
    while sobrante != 0 and restos:
        k = restos[i % len(restos)]
        if sobrante > 0:
            enteros[k] += 1
            sobrante -= 1
        elif enteros[k] > 1:
            enteros[k] -= 1
            sobrante += 1
        i += 1
        if i > 1000:
            break
    return enteros


@lru_cache(maxsize=16)
def curva_decision(modelo: str) -> pd.DataFrame:
    """Beneficio neto (Vickers y Elkin, 2006) en función del umbral.

    BN(t) = VP/n − FP/n · t/(1 − t). El umbral t expresa cuánto vale evitar
    una readmisión frente al costo de intervenir a quien no la tendría: con
    t = 0,2 se acepta intervenir a 4 pacientes sin riesgo por cada
    readmisión detectada. Se compara con «intervenir a todos» y «a nadie».
    """
    y, p = _yp(modelo)
    n, prevalencia = len(y), y.mean()
    filas = []
    for t in np.round(np.arange(0.02, 0.51, 0.01), 2):
        marcado = p >= t
        vp = (marcado & (y == 1)).sum()
        fp = (marcado & (y == 0)).sum()
        cociente = t / (1 - t)
        filas.append({"umbral": t, "modelo": vp / n - fp / n * cociente,
                      "todos": prevalencia - (1 - prevalencia) * cociente, "ninguno": 0.0})
    return pd.DataFrame(filas)


@lru_cache(maxsize=1)
def _mitades_prueba() -> np.ndarray:
    """Máscara de la mitad de calibración del conjunto de prueba, por paciente.

    La predicción conforme necesita datos que el modelo no vio. Los modelos
    se reentrenaron con todo el entrenamiento, así que se usa la prueba: la
    mitad de sus pacientes calibra y la otra mitad verifica la cobertura.
    Se parte por paciente para que nadie quede en las dos mitades.
    """
    prueba = D.prueba()
    ident = D.roles()["identificador"]
    pacientes = prueba[ident].unique() if ident in prueba else np.arange(len(prueba))
    aleatorio = np.random.default_rng(42)
    calibracion = set(aleatorio.choice(pacientes, size=len(pacientes) // 2, replace=False).tolist())
    if ident in prueba:
        return prueba[ident].isin(calibracion).to_numpy()
    return np.isin(np.arange(len(prueba)), list(calibracion))


def _conjunto(p: np.ndarray, q0: float, q1: float):
    """Conjunto conforme para probabilidades ``p`` (de la clase 1)."""
    contiene_0 = (p <= q0)        # puntaje de no conformidad de la clase 0: p
    contiene_1 = (1 - p <= q1)    # de la clase 1: 1 − p
    return contiene_0, contiene_1


@lru_cache(maxsize=16)
def conforme(modelo: str, alfa: float = 0.1) -> dict:
    """Predicción conforme de Mondrian (condicionada a la clase).

    Con puntaje s = 1 − p(clase verdadera), el cuantil de cada clase se fija
    en la mitad de calibración con la corrección de muestra finita
    ⌈(n_c + 1)(1 − α)⌉ / n_c. Así la cobertura de 1 − α se garantiza por
    separado para readmitidos y no readmitidos (con la versión marginal, los
    readmitidos, que son la minoría, quedarían por debajo).
    """
    y, p = _yp(modelo)
    cal = _mitades_prueba()
    cuantiles = {}
    for clase in (0, 1):
        puntajes = (p if clase == 0 else 1 - p)[cal & (y == clase)]
        n_c = len(puntajes)
        nivel = min(1.0, np.ceil((n_c + 1) * (1 - alfa)) / n_c)
        cuantiles[clase] = float(np.quantile(puntajes, nivel, method="higher"))
    ev = ~cal
    c0, c1 = _conjunto(p[ev], cuantiles[0], cuantiles[1])
    yev = y[ev]
    cubre = np.where(yev == 1, c1, c0)
    return {"alfa": alfa, "q0": cuantiles[0], "q1": cuantiles[1],
            "cobertura": float(cubre.mean()),
            "cobertura_0": float(cubre[yev == 0].mean()), "cobertura_1": float(cubre[yev == 1].mean()),
            "ambos": float((c0 & c1).mean()), "vacio": float((~c0 & ~c1).mean()),
            "n_calibracion": int(cal.sum()), "n_evaluacion": int(ev.sum())}


def conjunto_caso(modelo: str, p: float) -> tuple[bool, bool]:
    info = conforme(modelo)
    c0, c1 = _conjunto(np.array([p]), info["q0"], info["q1"])
    return bool(c0[0]), bool(c1[0])


#: Variables del parecido entre pacientes y su peso. La distancia es la de
#: Gower: diferencia absoluta escalada por el rango (p1–p99 del
#: entrenamiento) en las numéricas, 0/1 en las categóricas.
PARECIDO_NUMERICAS = ["age_ord", "time_in_hospital", "num_lab_procedures", "num_procedures",
                      "num_medications", "number_outpatient", "number_emergency",
                      "number_inpatient", "number_diagnoses"]
PARECIDO_CATEGORICAS = ["diag_1_grupo", "admission_type_id", "discharge_disposition_id",
                        "A1Cresult", "insulin", "change", "gender"]


@lru_cache(maxsize=1)
def _escalas_parecido() -> pd.Series:
    train = D.entrenamiento()
    rango = train[PARECIDO_NUMERICAS].quantile(0.99) - train[PARECIDO_NUMERICAS].quantile(0.01)
    return rango.replace(0, 1.0)


def pacientes_parecidos(caso: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Los ``k`` encuentros de prueba más parecidos al caso (distancia de Gower).

    Se busca en la prueba, no en el entrenamiento: el desenlace que se
    muestra es el real de pacientes que ningún modelo vio al ajustarse.
    """
    prueba = D.prueba()
    escalas = _escalas_parecido()
    fila = caso.iloc[0]
    numerica = ((prueba[PARECIDO_NUMERICAS] - fila[PARECIDO_NUMERICAS].astype(float)).abs()
                / escalas).clip(upper=1.0)
    categorica = pd.DataFrame({c: (prueba[c].astype(str) != str(fila[c])).astype(float)
                               for c in PARECIDO_CATEGORICAS})
    distancia = (numerica.sum(axis=1) + categorica.sum(axis=1)) / (
        len(PARECIDO_NUMERICAS) + len(PARECIDO_CATEGORICAS))
    cercanos = distancia.nsmallest(k).index
    salida = prueba.loc[cercanos].copy()
    salida["distancia"] = distancia.loc[cercanos]
    return salida


#: Atributos de la auditoría de equidad.
ATRIBUTOS_EQUIDAD = {"edad": "Edad", "gender": "Sexo", "race": "Grupo racial"}


def _grupo_edad(edad: pd.Series) -> pd.Series:
    """Tres grandes grupos de edad: los decenios por separado dejan grupos
    de pocas decenas de pacientes en prueba, con métricas muy inestables."""
    return pd.cut(edad, bins=[-1, 4, 6, 9], labels=["Menos de 50", "50 a 69", "70 o más"]).astype(str)


@lru_cache(maxsize=32)
def equidad(modelo: str, atributo: str) -> pd.DataFrame:
    """Desempeño del modelo por subgrupo en prueba, con el umbral operativo."""
    prueba = D.prueba()
    y, p = _yp(modelo)
    umbral = D.metadata(modelo)["umbral"]
    if atributo == "edad":
        grupos = _grupo_edad(prueba["age_ord"])
    else:
        grupos = prueba[atributo].astype(str).map(lambda v: C.nivel(atributo, v))
    filas = []
    for nombre_g in grupos.unique():
        m = (grupos == nombre_g).to_numpy()
        yg, pg = y[m], p[m]
        positivos = int(yg.sum())
        fila = {"grupo": nombre_g, "n": int(m.sum()), "readmitidos": positivos,
                "prevalencia": 100 * yg.mean(), "marcados": 100 * (pg >= umbral).mean(),
                "deteccion": 100 * ((pg >= umbral) & (yg == 1)).sum() / positivos if positivos else np.nan,
                "auc_roc": np.nan, "auc_pr": np.nan}
        if 0 < positivos < len(yg):
            fila["auc_roc"] = roc_auc_score(yg, pg)
            fila["auc_pr"] = average_precision_score(yg, pg)
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    tabla["fiable"] = tabla["readmitidos"] >= 30
    return tabla.sort_values("n", ascending=False)


def matriz_comparacion() -> tuple[pd.DataFrame, bool]:
    """Matriz fold × modelo de la comparación estadística.

    Usa la del protocolo (presupuesto 60, con el modelo original y los dos
    benchmarks del estado del arte) si está; si no, la del Entregable 2.
    El booleano dice cuál se usó, para que los textos lo indiquen.
    """
    protocolo = D.por_fold_protocolo()
    if protocolo is not None:
        return protocolo, True
    return D.por_fold_mejores(), False


#: Familias de la tabla a igual presupuesto, en el orden en que se explican.
FAMILIAS_PRESUPUESTO = ("Modelo base", "Estado del arte", "Ensamble estático", "Modelo original")


def por_fold_igual_presupuesto(evaluacion: pd.DataFrame, sota: pd.DataFrame | None = None) -> pd.DataFrame:
    """AUC-PR, AUC-ROC y Brier por fold de cada método a 60 evaluaciones.

    Formato largo: ``clave`` (la del tablero), ``familia``, ``fold``,
    ``auc_pr``, ``auc_roc`` y ``brier``. Deja fuera las ablaciones y la
    versión inicial, que no son competidores. Es la base de la tabla de
    líderes y de los pasos 3 a 5 de «Modelo original».
    """
    base = evaluacion[evaluacion["familia"].isin(["Modelo base", "Ensamble estático", "Modelo original"])]
    partes = [base.rename(columns={"método": "clave"})[["clave", "familia", "fold", "auc_pr", "auc_roc", "brier"]]]
    if sota is not None and len(sota):
        externo = sota.rename(columns={"modelo": "clave"})[["clave", "fold", "auc_pr", "auc_roc", "brier"]]
        partes.append(externo.assign(familia="Estado del arte"))
    por_fold = pd.concat(partes, ignore_index=True)
    # El nombre del modelo original en el CSV es «DES-CPL»; su clave del tablero es C.MODELO_ORIGINAL.
    por_fold["clave"] = por_fold["clave"].replace({"DES-CPL": C.MODELO_ORIGINAL})
    return por_fold


def tabla_igual_presupuesto(evaluacion: pd.DataFrame, sota: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla de líderes con todos los métodos a 60 evaluaciones.

    ``evaluacion`` es results/des_cpl_evaluacion_60.csv (fold, método, familia,
    auc_pr, auc_roc, brier) y ``sota`` results/benchmarks_sota.csv ya filtrado
    a 60 evaluaciones (modelo, fold, auc_pr, auc_roc, brier). Incluye los 7
    modelos base, los dos del estado del arte (si están), los tres ensambles
    estáticos y DES-CPL; deja fuera las ablaciones y la versión inicial, que
    no son competidores. Media y desviación (n - 1) sobre los folds externos.
    Orden: AUC-PR media, de mayor a menor.
    """
    por_fold = por_fold_igual_presupuesto(evaluacion, sota)
    agregado = por_fold.groupby("clave", sort=False).agg(
        familia=("familia", "first"), n_folds=("fold", "nunique"),
        auc_pr_media=("auc_pr", "mean"), auc_pr_sd=("auc_pr", "std"),
        auc_roc_media=("auc_roc", "mean"), auc_roc_sd=("auc_roc", "std"),
        brier_media=("brier", "mean")).reset_index()
    agregado["modelo"] = agregado["clave"].map(C.nombre_modelo)
    agregado["es_original"] = agregado["clave"] == C.MODELO_ORIGINAL
    return agregado.sort_values("auc_pr_media", ascending=False).reset_index(drop=True)


#: Margen de equivalencia en AUC-PR (la franja ±0,005 del protocolo).
EQUIVALENCIA_AUC_PR = 0.005


def lectura_lideres_presupuesto(tabla: pd.DataFrame) -> list[str]:
    """Frases de la tabla a igual presupuesto, calculadas con ella."""
    mejor = tabla.iloc[0]
    lineas = [f"A igual presupuesto, el mejor es {mejor['modelo']} con AUC-PR de {num(mejor['auc_pr_media'], 4)} "
              f"(± {num(mejor['auc_pr_sd'], 4)} entre folds)."]
    dentro = tabla[mejor["auc_pr_media"] - tabla["auc_pr_media"] <= EQUIVALENCIA_AUC_PR]
    lineas.append(f"{len(dentro)} de {len(tabla)} métodos quedan a menos de {num(EQUIVALENCIA_AUC_PR, 3)} de AUC-PR del "
                  f"mejor, y su desviación entre folds ({num(dentro['auc_pr_sd'].min(), 4)} a "
                  f"{num(dentro['auc_pr_sd'].max(), 4)}) es mayor que esa distancia: el orden del extremo "
                  "superior no es concluyente.")
    original = tabla[tabla["es_original"]]
    if len(original) and not original.index[0] == 0:
        o = original.iloc[0]
        distancia = mejor["auc_pr_media"] - o["auc_pr_media"]
        desviacion = min(o["auc_pr_sd"], mejor["auc_pr_sd"])
        # No se habla de «empate»: estar cerca no demuestra equivalencia. El paso 3
        # de /modelo-original clasifica el intervalo de esta diferencia como «sin
        # conclusión» (cruza el 0 y se sale de la franja de ±0,005).
        if distancia < desviacion:
            veredicto = (f"una distancia menor que la desviación entre folds ({num(o['auc_pr_sd'], 4)} y "
                         f"{num(mejor['auc_pr_sd'], 4)}): no se distinguen, sin conclusión en ningún sentido.")
        else:
            veredicto = (f"una distancia mayor que la desviación entre folds ({num(o['auc_pr_sd'], 4)} y "
                         f"{num(mejor['auc_pr_sd'], 4)}).")
        lineas.append(f"{o['modelo']} es el puesto {original.index[0] + 1} de {len(tabla)}, a "
                      f"{num(distancia, 4)} de {mejor['modelo']}: {veredicto}")
    elif len(original):
        lineas.append(f"{original.iloc[0]['modelo']} encabeza la tabla, pero la diferencia con el resto es menor "
                      "que la desviación entre folds.")
    return lineas


def lideres_igual_presupuesto() -> pd.DataFrame | None:
    """La tabla anterior con los CSV del repositorio, o None si falta la
    evaluación a 60 (sin LightGBM ni CatBoost si solo falta su CSV)."""
    evaluacion = D.evaluacion_60()
    if evaluacion is None:
        return None
    return tabla_igual_presupuesto(evaluacion, D.benchmarks_sota())


def rangos_por_fold() -> pd.DataFrame:
    """Rango de cada modelo en cada fold externo (1 = mejor AUC-PR)."""
    matriz, _ = matriz_comparacion()
    rangos = matriz.rank(axis=1, ascending=False)
    rangos.index = [f"Fold {i + 1}" for i in range(len(rangos))]
    return rangos


# =========================================================================== #
# Vistas que sustituyen al Sankey, al sunburst y a las categorías paralelas
# =========================================================================== #
#: Etapas del recorrido: (columna, título, icono de Material Symbols).
ETAPAS = {
    "admision": ("admission_type_id", "Cómo ingresó", "ambulance"),
    "estancia": ("time_in_hospital", "Cuánto tiempo estuvo", "bed"),
    "alta": ("discharge_disposition_id", "A dónde fue al salir", "home_health"),
}
CUALQUIERA = "__todos__"


def _etapas(df: pd.DataFrame, minimo: int = 200) -> pd.DataFrame:
    """Cada encuentro con la etiqueta legible de sus tres etapas.

    Los niveles con menos de ``minimo`` encuentros en todo el entrenamiento
    se agrupan en «Otro»: así las opciones del selector son estables (no
    aparecen ni desaparecen al mover los filtros) y cada una tiene una tasa
    fiable.
    """
    salida = pd.DataFrame({
        "admision": df["admission_type_id"].astype(str).map(lambda v: C.nivel("admission_type_id", v)),
        "estancia": _estancia(df["time_in_hospital"]).str.replace(r" \(", " (", regex=True),
        "alta": df["discharge_disposition_id"].astype(str).map(lambda v: C.nivel("discharge_disposition_id", v)),
        Y: df[Y].to_numpy(),
    }, index=df.index)
    for etapa in ("admision", "alta"):
        validos = set(niveles_etapa(etapa, minimo))
        salida.loc[~salida[etapa].isin(validos), etapa] = "Otro"
    return salida


@lru_cache(maxsize=4)
def niveles_etapa(etapa: str, minimo: int = 200) -> tuple:
    """Niveles de una etapa con al menos ``minimo`` encuentros, de más a menos frecuente."""
    if etapa == "estancia":
        return tuple(e[2] for e in ESTANCIA)
    columna = ETAPAS[etapa][0]
    serie = D.entrenamiento()[columna].astype(str).map(lambda v: C.nivel(columna, v))
    conteo = serie.value_counts()
    niveles = [n for n in conteo.index if conteo[n] >= minimo and n != "Otro"]
    return tuple(niveles + ["Otro"])


def resumen_etapas(df: pd.DataFrame) -> dict:
    """Tasa (con IC) de cada nivel de cada etapa, en el orden del selector."""
    datos = _etapas(df)
    salida = {}
    for etapa in ETAPAS:
        tabla = resumen_tasa(datos, etapa)
        orden = {n: i for i, n in enumerate(niveles_etapa(etapa))}
        tabla["orden"] = tabla[etapa].map(orden).fillna(99)
        salida[etapa] = tabla.sort_values("orden").rename(columns={etapa: "nivel"})
    return salida


def tasa_recorrido(df: pd.DataFrame, eleccion: dict) -> dict:
    """Encuentros y readmisión de los que siguieron exactamente ese recorrido.

    ``eleccion`` asigna a cada etapa un nivel o ``CUALQUIERA``.
    """
    datos = _etapas(df)
    mascara = np.ones(len(datos), dtype=bool)
    for etapa, nivel in eleccion.items():
        if nivel and nivel != CUALQUIERA:
            mascara &= (datos[etapa] == nivel).to_numpy()
    sub = datos[mascara]
    n, k = len(sub), int(sub[Y].sum())
    bajo, alto = wilson([k], [n]) if n else (np.array([np.nan]), np.array([np.nan]))
    return {"n": n, "k": k, "tasa": 100 * k / n if n else np.nan,
            "ic_bajo": 100 * float(bajo[0]), "ic_alto": 100 * float(alto[0])}


def ranking_diagnosticos(df: pd.DataFrame, columna: str = "diag_1_grupo", minimo: int = 100) -> pd.DataFrame:
    """Tasa (con IC) por capítulo diagnóstico, de mayor a menor, con al menos ``minimo`` encuentros."""
    datos = pd.DataFrame({"nivel": df[columna].astype(str), Y: df[Y].to_numpy()})
    tabla = resumen_tasa(datos, "nivel")
    return tabla[tabla["n"] >= minimo].sort_values("tasa", ascending=False)


def secundarios(df: pd.DataFrame, principal: str, minimo: int = 50) -> pd.DataFrame:
    """Ranking de diagnósticos secundarios dentro de un diagnóstico principal."""
    return ranking_diagnosticos(df[df["diag_1_grupo"].astype(str) == principal], "diag_2_grupo", minimo)


#: Variables de la matriz de riesgo y cómo se agrupan para leerse en una celda.
MATRIZ = ["number_inpatient", "age_ord", "time_in_hospital", "admission_type_id", "discharge_disposition_id",
          "insulin", "change", "A1Cresult", "diag_1_grupo", "gender", "number_emergency"]


def etiquetar(df: pd.DataFrame, variable: str, maximo: int = 7) -> pd.Series:
    """Nivel legible y con pocas categorías (las raras van a «Otros»)."""
    if variable == "age_ord":
        grupos = pd.cut(df[variable], bins=[-1, 4, 5, 6, 7, 9],
                        labels=["< 50", "50–59", "60–69", "70–79", "≥ 80"])
        return grupos.astype(str)
    if variable in ("number_inpatient", "number_emergency"):
        return df[variable].clip(upper=3).map(lambda v: "≥ 3" if v >= 3 else str(int(v)))
    if variable == "time_in_hospital":
        return _estancia(df[variable]).str.replace(r"Estancia (\w+) \((.*)\)", r"\2", regex=True)
    serie = df[variable].astype(str).map(lambda v: C.nivel(variable, v))
    frecuentes = serie.value_counts().index[: maximo - 1]
    return serie.where(serie.isin(frecuentes), "Otros")


ORDEN_NATURAL = {"age_ord": ["< 50", "50–59", "60–69", "70–79", "≥ 80"],
                 "number_inpatient": ["0", "1", "2", "≥ 3"], "number_emergency": ["0", "1", "2", "≥ 3"],
                 "time_in_hospital": ["1–3 días", "4–7 días", "8–14 días"],
                 "A1Cresult": ["No medida", "Normal", "> 7 %", "> 8 %"],
                 "insulin": ["No prescrito", "Dosis estable", "Dosis baja", "Dosis sube"]}


def matriz_riesgo(df: pd.DataFrame, fila: str, columna: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tasa de readmisión (%) y encuentros para cada combinación de dos variables."""
    datos = pd.DataFrame({"f": etiquetar(df, fila), "c": etiquetar(df, columna), Y: df[Y].to_numpy()})
    agregado = datos.groupby(["f", "c"], observed=True)[Y].agg(n="size", k="sum").reset_index()
    n = agregado.pivot(index="f", columns="c", values="n").fillna(0)
    k = agregado.pivot(index="f", columns="c", values="k").fillna(0)

    def ordenar(niveles, variable, conteo):
        natural = ORDEN_NATURAL.get(variable)
        if natural:
            return [v for v in natural if v in niveles] + [v for v in niveles if v not in natural]
        return sorted(niveles, key=lambda v: (v == "Otros", -conteo.get(v, 0)))
    filas = ordenar(list(n.index), fila, n.sum(axis=1).to_dict())
    columnas = ordenar(list(n.columns), columna, n.sum(axis=0).to_dict())
    n, k = n.loc[filas, columnas], k.loc[filas, columnas]
    tasa = 100 * k / n.where(n > 0)
    return tasa, n
