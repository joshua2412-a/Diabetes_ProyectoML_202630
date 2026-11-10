"""
Figuras del dashboard. Las páginas no estilan gráficos: los piden aquí.

La forma sigue al trabajo del dato: magnitud -> barras de un tono; identidad
de la clase -> su color; polaridad -> divergente con gris en el cero. Nunca
dos ejes Y: cuando hay dos medidas (distribución y tasa) van en paneles
separados con el eje X compartido. Toda figura termina en ``tematizar``.

Las funciones de figura que acompañan un hallazgo devuelven también las
frases de su lectura, calculadas con los mismos datos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from . import config as C
from . import datos as D
from .formato import miles, num, pct
from .tema import (CATEGORICA, CLASE, COLOR_MARCA, COLOR_TENUE, DIVERGENTE,
                   EJE, FONDO, RAMPA, TINTA, TINTA_TENUE, tematizar)

Y = "readmitted_30"
NINGUNO = "__ninguno__"
TASA = "__tasa__"


def _etiqueta_valor(variable, valor):
    """Valor legible: el grupo de edad en vez de su código, Sí/No en binarias."""
    if isinstance(valor, pd.Interval):
        # Variables enteras en intervalos (a, b]: contienen de a+1 a b.
        desde = int(np.floor(valor.left)) + 1
        if np.isinf(valor.right):
            return f"≥ {desde}"
        hasta = int(np.floor(valor.right))
        return str(desde) if hasta == desde else f"{desde}–{hasta}"
    if variable == "age_ord":
        return D.etiquetas_edad().get(int(valor), str(valor))
    if variable in D.roles()["binarias"]:
        return "Sí" if int(valor) == 1 else "No"
    if isinstance(valor, float) and valor.is_integer():
        return str(int(valor))
    return str(valor)


def _vacio(mensaje="No hay encuentros con estos filtros."):
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False, font=dict(color=TINTA_TENUE, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return tematizar(fig, alto=260)


def _agrupar_numerica(serie: pd.Series, maximo: int = 16) -> pd.Series:
    """Agrupa las variables con muchos valores, como categoría ordenada.

    Los conteos muy sesgados (hospitalizaciones o urgencias previas, con la
    mayoría en 0) conservan sus valores frecuentes uno a uno y agrupan la
    cola en «≥ k»: los deciles fundirían 0 y 1, justo la distinción que más
    importa. El resto se agrupa en deciles. La categoría ordenada hace que
    los grupos salgan en orden numérico y no alfabético.
    """
    if serie.nunique() <= maximo:
        return serie
    valores = serie.dropna()
    if (valores % 1 == 0).all():
        minimo, tope = int(valores.min()), int(valores.quantile(0.95))
        if tope - minimo + 1 <= maximo - 1:
            bordes = list(np.arange(minimo - 0.5, tope + 1.0, 1.0)) + [np.inf]
            return pd.cut(serie, bins=bordes)
    return pd.qcut(serie, q=min(10, maximo), duplicates="drop")


# =========================================================================== #
# EDA
# =========================================================================== #
def fig_objetivo(df: pd.DataFrame):
    if df.empty:
        return _vacio(), []
    conteo = df[Y].value_counts().reindex([0, 1]).fillna(0)
    total = conteo.sum()
    fig = go.Figure()
    for clase in [0, 1]:
        fig.add_trace(go.Bar(
            y=[C.CLASES_CORTAS[clase]], x=[conteo[clase]], orientation="h",
            name=C.CLASES[clase], marker=dict(color=CLASE[clase]),
            text=[f"{miles(conteo[clase])} ({pct(100 * conteo[clase] / total)})"],
            textposition="outside", cliponaxis=False,
            hovertemplate=f"{C.CLASES[clase]}: %{{x:,}} encuentros<extra></extra>"))
    fig.update_xaxes(title_text="Encuentros", showgrid=True, gridcolor="#ECEEF6")
    fig.update_yaxes(autorange="reversed")
    tasa = 100 * conteo[1] / total
    lectura = [
        f"El {pct(tasa)} de los {miles(total)} encuentros filtrados termina en "
        f"una readmisión antes de 30 días: la clase de interés es la minoritaria.",
        "Con ese desbalance, la exactitud engaña: un modelo que nunca predice "
        f"readmisión acierta el {pct(100 - tasa)}. Por eso los modelos se comparan con AUC-PR.",
    ]
    return tematizar(fig, alto=220, showlegend=False, bargap=0.45,
                     margin=dict(l=110, r=110, t=20, b=40)), lectura


def fig_univariado(df: pd.DataFrame, variable: str):
    """Distribución por clase (arriba) y tasa de readmisión (abajo)."""
    if df.empty:
        return _vacio(), []
    datos = df[[variable, Y]].copy()
    numerica = D.es_numerica(variable)
    if numerica:
        datos["nivel"] = _agrupar_numerica(datos[variable])
    else:
        datos["nivel"] = datos[variable].astype(str)
    resumen = datos.groupby("nivel", observed=True)[Y].agg(["size", "sum", "mean"])
    if not numerica:
        resumen = resumen.sort_values("mean", ascending=False)
    niveles = list(resumen.index)
    etiquetas = [_etiqueta_valor(variable, v) if numerica else str(v) for v in niveles]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.58, 0.42],
                        vertical_spacing=0.08)
    for clase, columna in [(0, None), (1, "sum")]:
        valores = resumen["size"] - resumen["sum"] if clase == 0 else resumen["sum"]
        fig.add_trace(go.Bar(x=etiquetas, y=valores, name=C.CLASES[clase],
                             marker=dict(color=CLASE[clase], line=dict(color=FONDO, width=1)),
                             hovertemplate="%{x}: %{y:,} encuentros<extra>" + C.CLASES[clase] + "</extra>"),
                      row=1, col=1)
    prevalencia = 100 * df[Y].mean()
    fig.add_trace(go.Scatter(x=etiquetas, y=100 * resumen["mean"], mode="lines+markers",
                             name="Tasa de readmisión", line=dict(color=COLOR_MARCA, width=2),
                             marker=dict(size=8, color=COLOR_MARCA, line=dict(color=FONDO, width=2)),
                             customdata=resumen["size"],
                             hovertemplate="%{x}: %{y:.1f} % (n = %{customdata:,})<extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=prevalencia, row=2, col=1, line=dict(color=TINTA_TENUE, dash="dot", width=1),
                  annotation_text=f"media {pct(prevalencia)}", annotation_position="top left",
                  annotation_font=dict(size=11, color=TINTA_TENUE))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Encuentros", row=1, col=1)
    fig.update_yaxes(title_text="Readmisión (%)", row=2, col=1)
    fig.update_xaxes(title_text=C.nombre(variable), row=2, col=1, type="category")

    lectura = []
    fiables = resumen[resumen["size"] >= 50]
    if len(fiables) >= 2:
        alto, bajo = fiables["mean"].idxmax(), fiables["mean"].idxmin()
        lectura.append(
            f"La readmisión va del {pct(100 * fiables.loc[bajo, 'mean'])} en "
            f"«{_etiqueta_valor(variable, bajo) if numerica else bajo}» "
            f"al {pct(100 * fiables.loc[alto, 'mean'])} en "
            f"«{_etiqueta_valor(variable, alto) if numerica else alto}» "
            f"(grupos con al menos 50 encuentros; media {pct(prevalencia)}).")
    if numerica and df[variable].nunique() > 1:
        r = stats.spearmanr(df[variable], df[Y]).statistic
        lectura.append(f"Correlación de Spearman con la readmisión: $r = {num(r, 3)}$. "
                       "Una asociación débil por sí sola no descarta que la variable "
                       "aporte combinada con otras.")
    return tematizar(fig, alto=430, margin=dict(l=60, r=20, t=40, b=60)), lectura


def _colores_grupo(serie: pd.Series, variable: str | None = None):
    """Grupos para colorear y su color.

    Categóricas: los 7 niveles más frecuentes + «Otros». Numéricas y
    ordinales (la edad): grupos ordenados por valor, en el orden de la
    leyenda; con más de 8 valores se agrupan (la edad junta sus decenios
    menos frecuentes, los de menor edad).
    """
    if variable is not None and D.es_numerica(variable):
        valores = serie.dropna()
        niveles = sorted(valores.unique())
        if variable == "age_ord" and len(niveles) > 8:
            corte = niveles[len(niveles) - 8]
            primera = f"hasta {_etiqueta_valor(variable, corte)}"
            etiquetas = serie.map(lambda v: primera if v <= corte else _etiqueta_valor(variable, v))
            orden = [primera] + [_etiqueta_valor(variable, v) for v in niveles if v > corte]
        else:
            grupos = _agrupar_numerica(serie, maximo=8)
            ordenados = (list(grupos.cat.categories) if hasattr(grupos, "cat")
                         else sorted(grupos.dropna().unique()))
            etiquetas = grupos.map(lambda v: _etiqueta_valor(variable, v)).astype(str)
            orden = [_etiqueta_valor(variable, v) for v in ordenados]
        orden = [g for g in orden if g in set(etiquetas)]
        return etiquetas, {g: CATEGORICA[i] for i, g in enumerate(orden)}
    serie = serie.astype(str)
    frecuentes = serie.value_counts().index.tolist()
    if len(frecuentes) > 8:
        serie = serie.where(serie.isin(frecuentes[:7]), "Otros")
        frecuentes = frecuentes[:7] + ["Otros"]
    return serie, {g: CATEGORICA[i] for i, g in enumerate(frecuentes)}


def fig_bivariado(df: pd.DataFrame, x: str, y: str, color: str):
    if df.empty:
        return _vacio(), []
    datos = df.sample(min(len(df), 4000), random_state=42) if y != TASA else df
    if color == Y:
        grupos = datos[Y].map(C.CLASES_CORTAS)
        paleta = {C.CLASES_CORTAS[k]: v for k, v in CLASE.items()}
    elif color == NINGUNO:
        grupos = pd.Series("Todos", index=datos.index)
        paleta = {"Todos": COLOR_MARCA}
    else:
        grupos, paleta = _colores_grupo(datos[color], color)

    fig = go.Figure()
    lectura = []
    if y == TASA:
        nivel = _agrupar_numerica(datos[x]) if D.es_numerica(x) else datos[x].astype(str)
        tabla = (pd.DataFrame({"nivel": nivel, "grupo": grupos, Y: datos[Y]})
                 .groupby(["grupo", "nivel"], observed=True)[Y].agg(["size", "mean"]).reset_index())
        for grupo, color_g in paleta.items():
            sub = tabla[(tabla["grupo"] == grupo) & (tabla["size"] >= 20)]
            fig.add_trace(go.Scatter(
                x=[_etiqueta_valor(x, v) if D.es_numerica(x) else str(v) for v in sub["nivel"]],
                y=100 * sub["mean"], mode="lines+markers", name=str(grupo),
                line=dict(color=color_g, width=2), marker=dict(size=8, line=dict(color=FONDO, width=2)),
                customdata=sub["size"],
                hovertemplate="%{x}: %{y:.1f} % (n = %{customdata:,})<extra>" + str(grupo) + "</extra>"))
        fig.update_xaxes(type="category", title_text=C.nombre(x))
        fig.update_yaxes(title_text="Readmisión (%)")
        lectura.append("Cada punto es la tasa de readmisión del grupo; se omiten los grupos "
                       "con menos de 20 encuentros, cuya tasa es inestable.")
    elif D.es_numerica(x):
        aleatorio = np.random.default_rng(42)
        for grupo, color_g in paleta.items():
            sub = datos[grupos == grupo]
            # Las variables son enteras: un temblor pequeño separa los puntos
            # que caerían exactamente encima unos de otros.
            fig.add_trace(go.Scattergl(
                x=sub[x] + aleatorio.uniform(-0.3, 0.3, len(sub)),
                y=sub[y] + aleatorio.uniform(-0.3, 0.3, len(sub)),
                mode="markers", name=str(grupo),
                marker=dict(size=6, color=color_g, opacity=0.55, line=dict(color=FONDO, width=0.5)),
                hovertemplate=f"{C.nombre(x)}: %{{x:.0f}}<br>{C.nombre(y)}: %{{y:.0f}}<extra>{grupo}</extra>"))
        fig.update_xaxes(title_text=C.nombre(x))
        fig.update_yaxes(title_text=C.nombre(y))
        r = stats.spearmanr(datos[x], datos[y]).statistic
        lectura.append(f"Correlación de Spearman entre {C.nombre(x).lower()} y "
                       f"{C.nombre(y).lower()}: $r = {num(r, 3)}$ (muestra de "
                       f"{miles(len(datos))} encuentros).")
    else:
        for grupo, color_g in paleta.items():
            sub = datos[grupos == grupo]
            fig.add_trace(go.Box(x=sub[x].astype(str), y=sub[y], name=str(grupo),
                                 marker=dict(color=color_g), line=dict(width=1.5),
                                 boxpoints=False))
        fig.update_layout(boxmode="group")
        fig.update_xaxes(title_text=C.nombre(x), type="category")
        fig.update_yaxes(title_text=C.nombre(y))
        lectura.append("Cada caja resume la variable del eje Y en un nivel del eje X: la línea "
                       "central es la mediana y la caja va del primer al tercer cuartil.")
    fig.update_layout(showlegend=len(paleta) > 1)
    return tematizar(fig, alto=430), lectura


def fig_correlacion(df: pd.DataFrame):
    if len(df) < 10:
        return _vacio(), []
    r = D.roles()
    columnas = r["numericas"] + r["ordinales"] + r["binarias"] + [Y]
    matriz = df[columnas].corr(method="spearman")
    nombres = [C.nombre(c) for c in columnas]
    fig = go.Figure(go.Heatmap(
        z=matriz.values, x=nombres, y=nombres, zmin=-1, zmax=1, colorscale=DIVERGENTE,
        text=[[num(v, 2) for v in fila] for fila in matriz.values], texttemplate="%{text}",
        textfont=dict(size=10), colorbar=dict(title="r", thickness=12),
        hovertemplate="%{y} · %{x}: r = %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickangle=-40)
    con_y = matriz[Y].drop(Y)
    principal = con_y.abs().idxmax()
    pares = matriz.drop(index=Y, columns=Y).where(~np.eye(len(columnas) - 1, dtype=bool))
    a, b = pares.abs().stack().idxmax()
    lectura = [
        f"La variable más asociada con la readmisión es {C.nombre(principal).lower()} "
        f"($r = {num(con_y[principal], 3)}$); todas las correlaciones con el desenlace son débiles.",
        f"El par de predictoras más redundante es {C.nombre(a).lower()} y "
        f"{C.nombre(b).lower()} ($r = {num(matriz.loc[a, b], 2)}$).",
    ]
    return tematizar(fig, alto=560, margin=dict(l=170, r=20, t=20, b=150)), lectura


def fig_faltantes(tabla: pd.DataFrame):
    tabla = tabla.sort_values("porcentaje")
    fig = go.Figure(go.Bar(
        x=tabla["porcentaje"], y=tabla["variable"], orientation="h",
        marker=dict(color=COLOR_MARCA), customdata=tabla[["faltantes", "decision"]],
        text=[pct(p) for p in tabla["porcentaje"]], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f} % (%{customdata[0]:,} registros)<br>%{customdata[1]}<extra></extra>"))
    fig.update_xaxes(title_text="Registros sin dato (%)", range=[0, 110], showgrid=True, gridcolor="#ECEEF6")
    return tematizar(fig, alto=300, showlegend=False, margin=dict(l=140, r=60, t=20, b=45))


def fig_atipicos(df: pd.DataFrame):
    if df.empty:
        return _vacio(), []
    columnas = D.roles()["numericas"]
    filas = int(np.ceil(len(columnas) / 3))
    fig = make_subplots(rows=filas, cols=3, subplot_titles=[C.nombre(c) for c in columnas],
                        vertical_spacing=0.12, horizontal_spacing=0.08)
    conteo = {}
    for i, columna in enumerate(columnas):
        serie = df[columna]
        q1, q3 = serie.quantile([0.25, 0.75])
        rango = q3 - q1
        conteo[columna] = float(((serie < q1 - 1.5 * rango) | (serie > q3 + 1.5 * rango)).mean() * 100)
        fig.add_trace(go.Box(x=serie, name=C.nombre(columna), orientation="h",
                             marker=dict(color=COLOR_MARCA), line=dict(width=1.5),
                             boxpoints=False, hoverinfo="x"),
                      row=i // 3 + 1, col=i % 3 + 1)
    fig.update_yaxes(showticklabels=False)
    fig.update_annotations(font=dict(size=12, color=TINTA))
    mas = max(conteo, key=conteo.get)
    lectura = [
        f"Con la regla de 1,5 veces el rango intercuartílico, {C.nombre(mas).lower()} "
        f"tiene la mayor proporción de atípicos ({pct(conteo[mas])} de los encuentros).",
        "No se recortan: los valores extremos de la utilización previa son los encuentros "
        "más informativos, no errores de captura. El escalado robusto (mediana y rango "
        "intercuartílico) los tolera sin distorsionar el resto (capítulo 3).",
    ]
    return tematizar(fig, alto=170 * filas + 60, showlegend=False,
                     margin=dict(l=20, r=20, t=40, b=30)), lectura, conteo


# =========================================================================== #
# Modelos
# =========================================================================== #
def fig_lideres(tabla: pd.DataFrame, prevalencia: float):
    tabla = tabla.sort_values("auc_pr_media")
    nombres = [C.MODELOS.get(m, m) for m in tabla["modelo"]]
    colores = [COLOR_MARCA if f else COLOR_TENUE for f in tabla["finalista"]]
    fig = go.Figure(go.Scatter(
        x=tabla["auc_pr_media"], y=nombres, mode="markers",
        error_x=dict(type="data", array=tabla["auc_pr_sd"], color=EJE, thickness=2, width=4),
        marker=dict(size=12, color=colores, line=dict(color=FONDO, width=2)),
        customdata=np.column_stack([tabla["auc_pr_sd"], tabla["balanceo"], tabla["optimizador"]]),
        hovertemplate="%{y}: %{x:.4f} ± %{customdata[0]:.4f}<br>%{customdata[1]} + %{customdata[2]}<extra></extra>"))
    fig.add_vline(x=prevalencia, line=dict(color=TINTA_TENUE, dash="dot", width=1),
                  annotation_text=f"azar: {num(prevalencia, 3)}", annotation_position="bottom right",
                  annotation_font=dict(size=11, color=TINTA_TENUE))
    fig.update_xaxes(title_text="AUC-PR del bucle externo (media ± desviación, 5 folds)",
                     showgrid=True, gridcolor="#ECEEF6")
    return tematizar(fig, alto=340, showlegend=False, margin=dict(l=150, r=30, t=20, b=50))


def fig_confusion(y, p, umbral: float):
    predicho = (np.asarray(p) >= umbral).astype(int)
    y = np.asarray(y)
    matriz = np.array([[((y == r) & (predicho == c)).sum() for c in (0, 1)] for r in (0, 1)])
    porcentaje = 100 * matriz / matriz.sum(axis=1, keepdims=True)
    etiquetas = [C.CLASES_CORTAS[0], C.CLASES_CORTAS[1]]
    fig = go.Figure(go.Heatmap(
        z=porcentaje, x=etiquetas, y=etiquetas, colorscale=[[i / 6, c] for i, c in enumerate(RAMPA)],
        zmin=0, zmax=100, showscale=False,
        text=[[f"{miles(matriz[i, j])}<br>{pct(porcentaje[i, j])}" for j in range(2)] for i in range(2)],
        texttemplate="%{text}", textfont=dict(size=14),
        hovertemplate="Real: %{y}<br>Predicho: %{x}<br>%{z:.1f} % de la fila<extra></extra>"))
    fig.update_xaxes(title_text="Predicho", side="bottom")
    fig.update_yaxes(title_text="Real", autorange="reversed", showgrid=False)
    return tematizar(fig, alto=320, margin=dict(l=110, r=20, t=20, b=60))


def fig_roc(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=COLOR_MARCA, width=2),
                               name=f"Modelo (AUC {num(auc, 3)})",
                               hovertemplate="FPR %{x:.3f} · TPR %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                             line=dict(color=TINTA_TENUE, dash="dot", width=1), hoverinfo="skip"))
    fig.update_xaxes(title_text="Tasa de falsos positivos", range=[0, 1])
    fig.update_yaxes(title_text="Tasa de verdaderos positivos", range=[0, 1.02])
    return tematizar(fig, alto=320, margin=dict(l=60, r=20, t=40, b=50))


def fig_pr(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    base = float(np.mean(y))
    fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", line=dict(color=COLOR_MARCA, width=2),
                               name=f"Modelo (AP {num(ap, 3)})",
                               hovertemplate="Recall %{x:.3f} · Precisión %{y:.3f}<extra></extra>"))
    fig.add_hline(y=base, line=dict(color=TINTA_TENUE, dash="dot", width=1),
                  annotation_text=f"azar: {num(base, 3)}", annotation_position="top right",
                  annotation_font=dict(size=11, color=TINTA_TENUE))
    fig.update_xaxes(title_text="Recall (sensibilidad)", range=[0, 1])
    fig.update_yaxes(title_text="Precisión", range=[0, 1.02])
    return tematizar(fig, alto=320, margin=dict(l=60, r=20, t=40, b=50))


def fig_calibracion(y, p, bins: int = 10):
    datos = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    datos["bin"] = pd.qcut(datos["p"], q=bins, duplicates="drop")
    resumen = datos.groupby("bin", observed=True).agg(predicha=("p", "mean"),
                                                      observada=("y", "mean"), n=("y", "size"))
    tope = float(max(resumen["predicha"].max(), resumen["observada"].max()) * 1.1)
    fig = go.Figure(go.Scatter(x=[0, tope], y=[0, tope], mode="lines", name="Calibración perfecta",
                               line=dict(color=TINTA_TENUE, dash="dot", width=1), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=resumen["predicha"], y=resumen["observada"], mode="lines+markers",
                             name="Modelo", line=dict(color=COLOR_MARCA, width=2),
                             marker=dict(size=9, line=dict(color=FONDO, width=2)),
                             customdata=resumen["n"],
                             hovertemplate="Predicha %{x:.3f} · Observada %{y:.3f} (n = %{customdata:,})<extra></extra>"))
    ece = float((resumen["n"] * (resumen["predicha"] - resumen["observada"]).abs()).sum() / resumen["n"].sum())
    fig.update_xaxes(title_text="Probabilidad predicha (media del decil)", range=[0, tope])
    fig.update_yaxes(title_text="Frecuencia observada", range=[0, tope])
    return tematizar(fig, alto=320, margin=dict(l=60, r=20, t=40, b=50)), ece


def fig_shap_local(contribuciones: pd.Series, caso: pd.DataFrame, top: int = 10):
    """Las variables que más empujan la predicción del caso, con su signo."""
    orden = contribuciones.abs().sort_values(ascending=False)
    principales = contribuciones[orden.index[:top]]
    resto = contribuciones[orden.index[top:]].sum()
    etiquetas = [f"{C.nombre(v)} = {_etiqueta_valor(v, caso[v].iloc[0])}" for v in principales.index]
    valores = list(principales.values)
    if len(orden) > top:
        etiquetas.append(f"Otras {len(orden) - top} variables")
        valores.append(resto)
    colores = [CLASE[1] if v > 0 else CLASE[0] for v in valores]
    fig = go.Figure(go.Bar(
        x=[100 * v for v in valores], y=etiquetas, orientation="h",
        marker=dict(color=colores),
        text=[f"{num(100 * v, 1, signo=True)} pp" for v in valores], textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:+.2f} puntos porcentuales<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=EJE, width=1))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Aporte a la probabilidad de readmisión (puntos porcentuales)",
                     showgrid=True, gridcolor="#ECEEF6")
    return tematizar(fig, alto=60 + 30 * len(etiquetas), showlegend=False,
                     margin=dict(l=260, r=60, t=10, b=50))


def fig_cohorte(p_prueba, y_prueba, p_caso: float, umbral: float):
    """Dónde cae el caso entre los pacientes de prueba, por su desenlace real."""
    fig = go.Figure()
    for clase in [0, 1]:
        fig.add_trace(go.Histogram(
            x=np.asarray(p_prueba)[np.asarray(y_prueba) == clase], name=C.CLASES[clase],
            histnorm="percent", xbins=dict(start=0, end=1, size=0.02), opacity=0.75,
            marker=dict(color=CLASE[clase], line=dict(color=FONDO, width=1)),
            hovertemplate="P entre %{x}: %{y:.1f} %<extra>" + C.CLASES[clase] + "</extra>"))
    fig.add_vline(x=umbral, line=dict(color=TINTA_TENUE, dash="dot", width=1),
                  annotation_text="umbral", annotation_position="top left",
                  annotation_font=dict(size=11, color=TINTA_TENUE))
    fig.add_vline(x=p_caso, line=dict(color=TINTA, width=2.5),
                  annotation_text=f"este caso: {pct(100 * p_caso)}", annotation_position="top right",
                  annotation_font=dict(size=12, color=TINTA))
    tope = float(np.quantile(p_prueba, 0.995))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Probabilidad de readmisión predicha", tickformat=".0%",
                     range=[0, max(tope, p_caso) * 1.08])
    fig.update_yaxes(title_text="Pacientes de su grupo (%)", showgrid=True, gridcolor="#ECEEF6")
    return tematizar(fig, alto=300, margin=dict(l=60, r=20, t=40, b=50))


def fig_shap_global(tabla: pd.DataFrame, modelo: str, top: int = 15):
    datos = tabla[tabla["modelo"] == modelo]
    importancia = datos.assign(abs=datos["shap"].abs()).groupby("variable")["abs"].mean()
    importancia = importancia.sort_values(ascending=False).head(top)[::-1]
    fig = go.Figure(go.Bar(
        x=100 * importancia.values, y=[C.nombre(v) for v in importancia.index], orientation="h",
        marker=dict(color=COLOR_MARCA),
        hovertemplate="%{y}: %{x:.2f} pp de aporte medio<extra></extra>"))
    fig.update_xaxes(title_text="Aporte medio absoluto a la probabilidad (puntos porcentuales)",
                     showgrid=True, gridcolor="#ECEEF6")
    return tematizar(fig, alto=80 + 26 * len(importancia), showlegend=False,
                     margin=dict(l=210, r=30, t=10, b=50)), importancia[::-1]


def fig_diferencia_critica(matriz: pd.DataFrame, cd: float, destacado: str | None = None):
    """Diagrama de diferencia crítica de Nemenyi: modelos unidos por una barra
    no se distinguen estadísticamente."""
    rangos = matriz.rank(axis=1, ascending=False).mean().sort_values()
    nombres = [C.MODELOS.get(m, m) for m in rangos.index]
    fig = go.Figure(go.Scatter(
        x=rangos.values, y=[0] * len(rangos), mode="markers",
        marker=dict(size=12, color=[TINTA if m == destacado else COLOR_MARCA for m in rangos.index],
                    line=dict(color=FONDO, width=2)),
        text=nombres, hovertemplate="%{text}: rango medio %{x:.2f}<extra></extra>"))
    for i, (modelo, valor) in enumerate(rangos.items()):
        altura = -0.18 * (i + 1)
        fig.add_shape(type="line", x0=valor, x1=valor, y0=0, y1=altura,
                      line=dict(color=EJE, width=1))
        fig.add_annotation(x=valor, y=altura, text=f"<b>{nombres[i]}</b> ({num(valor, 2)})"
                           if modelo == destacado else f"{nombres[i]} ({num(valor, 2)})",
                           showarrow=False, yanchor="top", font=dict(size=12, color=TINTA))
    nivel = 0.12
    valores = rangos.values
    for i in range(len(valores)):
        juntos = [j for j in range(i, len(valores)) if valores[j] - valores[i] <= cd]
        if len(juntos) > 1 and (i == 0 or valores[juntos[-1]] - valores[i - 1] > cd):
            fig.add_shape(type="line", x0=valores[i] - 0.03, x1=valores[juntos[-1]] + 0.03,
                          y0=nivel, y1=nivel, line=dict(color=COLOR_MARCA, width=4))
            nivel += 0.09
    fig.add_shape(type="line", x0=valores.min(), x1=valores.min() + cd, y0=nivel + 0.1,
                  y1=nivel + 0.1, line=dict(color=TINTA, width=1.5))
    fig.add_annotation(x=valores.min() + cd / 2, y=nivel + 0.1, text=f"CD = {num(cd, 2)}",
                       showarrow=False, yanchor="bottom", font=dict(size=12, color=TINTA))
    fig.update_yaxes(visible=False, range=[-0.18 * (len(rangos) + 1.5), nivel + 0.3])
    fig.update_xaxes(title_text="Rango medio en AUC-PR (1 = mejor)", range=[0.5, len(rangos) + 0.5],
                     dtick=1)
    return tematizar(fig, alto=380, showlegend=False, margin=dict(l=30, r=30, t=20, b=50))
