"""
Figuras de las vistas clínicas (EDA) y de decisión (Modelos).

Mismo contrato que graficos.py: las páginas piden figuras aquí, cada figura
termina en ``tematizar`` y las que acompañan un hallazgo devuelven también
las frases de su lectura, calculadas con los mismos datos.

Color: el azul y el ámbar de las clases (no readmitido / readmitido) solo
aparecen cuando el dato habla del desenlace; la magnitud de una tasa usa la
rampa ámbar de un tono, y una serie sin clase usa COLOR_MARCA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from . import analitica as A
from . import config as C
from . import datos as D
from .formato import miles, num, pct
from .graficos import _vacio
from .tema import (AZUR, CLASE, COLOR_MARCA, COLOR_SEGUNDO, COLOR_TENUE, DIVERGENTE, EJE,
                   ESCALA_AMBAR, FONDO, MEDIANOCHE, REJILLA, TINTA, TINTA_TENUE, con_guia, rgba,
                   tematizar)

Y = A.Y


def _divergente_centrado(valores, centro: float, minimo_rango: float = 4.0):
    """Límites simétricos alrededor de la prevalencia para una escala divergente.

    Centrar en la prevalencia hace que el gris signifique «como la media» y
    que el ámbar y el azul signifiquen «por encima» y «por debajo».
    """
    valores = pd.Series(valores).dropna()
    if valores.empty:
        return centro - minimo_rango, centro + minimo_rango
    radio = max(minimo_rango, float((valores - centro).abs().quantile(0.95)))
    return centro - radio, centro + radio


# =========================================================================== #
# EDA clínico
# =========================================================================== #
def fig_recorrido(datos: dict, prevalencia: float):
    nodos, enlaces = datos["nodos"], datos["enlaces"]
    if nodos.empty or enlaces.empty:
        return _vacio(), []
    color_etapa = {"Admisión": MEDIANOCHE, "Estancia": "#5A85C4", "Alta": "#85A4D6"}
    colores = [CLASE[int(n == C.CLASES_CORTAS[1])] if e == "Desenlace" else color_etapa[e]
               for e, n in zip(nodos["etapa"], nodos["nivel"])]
    colores_enlace = [rgba(CLASE[int(c)], 0.45) if f else rgba("#85A4D6", 0.22)
                      for f, c in zip(enlaces["final"], enlaces["clase"].fillna(0))]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=[f"{n}" for n in nodos["nivel"]], color=colores, pad=14, thickness=16,
                  line=dict(color=FONDO, width=1),
                  customdata=np.column_stack([nodos["n"], nodos["tasa"], nodos["etapa"]]),
                  hovertemplate="<b>%{label}</b> (%{customdata[2]})<br>%{customdata[0]:,} encuentros"
                                "<br>Readmisión: %{customdata[1]:.1f} %<extra></extra>"),
        link=dict(source=enlaces["origen"], target=enlaces["destino"], value=enlaces["n"],
                  color=colores_enlace, customdata=enlaces["tasa"],
                  hovertemplate="%{source.label} → %{target.label}<br>%{value:,} encuentros"
                                "<br>Readmisión del flujo: %{customdata:.1f} %<extra></extra>"),
        textfont=dict(size=12, color=TINTA),
    ))
    for i, etapa in enumerate(["Tipo de admisión", "Estancia", "Destino al alta", "Desenlace"]):
        fig.add_annotation(x=i / 3, y=1.08, xref="paper", yref="paper", text=f"<b>{etapa}</b>",
                           showarrow=False, font=dict(size=12, color=TINTA_TENUE),
                           xanchor="left" if i == 0 else ("right" if i == 3 else "center"))
    altas = nodos[(nodos["etapa"] == "Alta") & (nodos["n"] >= 200)].sort_values("tasa")
    estancias = nodos[nodos["etapa"] == "Estancia"].sort_values("tasa")
    lectura = []
    if len(altas) >= 2:
        alto, bajo = altas.iloc[-1], altas.iloc[0]
        lectura.append(f"El destino al alta separa mucho el riesgo: «{alto['nivel']}» reingresa en el "
                       f"{pct(alto['tasa'])} de los casos y «{bajo['nivel']}», en el {pct(bajo['tasa'])} "
                       f"(media {pct(prevalencia)}).")
    if len(estancias) >= 2:
        lectura.append(f"Cuanto más larga la estancia, mayor la readmisión: de {pct(estancias.iloc[0]['tasa'])} "
                       f"en «{estancias.iloc[0]['nivel'].lower()}» a {pct(estancias.iloc[-1]['tasa'])} en "
                       f"«{estancias.iloc[-1]['nivel'].lower()}».")
    lectura.append("Pasa el ratón por un flujo para ver cuántos pacientes siguen ese camino y qué parte "
                   "vuelve antes de 30 días; arrastra los nodos para reordenarlos.")
    return tematizar(fig, alto=520, margin=dict(l=10, r=10, t=50, b=10)), lectura


def fig_diagnosticos(tabla: pd.DataFrame, prevalencia: float):
    if tabla.empty:
        return _vacio(), []
    cmin, cmax = _divergente_centrado(tabla["tasa"], prevalencia)
    fig = go.Figure(go.Sunburst(
        ids=tabla["id"], labels=tabla["etiqueta"], parents=tabla["padre"], values=tabla["n"],
        branchvalues="total", maxdepth=2, insidetextorientation="radial",
        marker=dict(colors=tabla["tasa"], colorscale=DIVERGENTE, cmin=cmin, cmax=cmax, cmid=prevalencia,
                    line=dict(color=FONDO, width=1.5),
                    colorbar=dict(title=dict(text="Readmisión (%)", side="right"), thickness=12,
                                  ticksuffix=" %")),
        customdata=np.column_stack([tabla["n"], tabla["tasa"]]),
        hovertemplate="<b>%{label}</b><br>%{customdata[0]:,} encuentros<br>"
                      "Readmisión: %{customdata[1]:.1f} %<extra></extra>",
    ))
    principales = tabla[(tabla["padre"] == "") & (tabla["n"] >= 300)].sort_values("tasa")
    lectura = []
    if len(principales) >= 2:
        alto, bajo = principales.iloc[-1], principales.iloc[0]
        lectura.append(f"Por diagnóstico principal, la readmisión va del {pct(bajo['tasa'])} "
                       f"(«{bajo['etiqueta']}») al {pct(alto['tasa'])} («{alto['etiqueta']}»).")
    grandes = principales.sort_values("n", ascending=False)
    if len(grandes):
        lectura.append(f"El capítulo más frecuente es «{grandes.iloc[0]['etiqueta']}»: "
                       f"{miles(grandes.iloc[0]['n'])} encuentros.")
    lectura.append("Ámbar: más readmisión que la media; azul: menos; gris: como la media. Haz clic en "
                   "un diagnóstico principal para ver sus secundarios y en el centro para volver.")
    fig.update_layout(uniformtext=dict(minsize=10, mode="hide"))
    return tematizar(fig, alto=540, margin=dict(l=10, r=10, t=10, b=10)), lectura


def fig_piramide(tabla: pd.DataFrame):
    if tabla.empty:
        return _vacio(), []
    edades = sorted(tabla["age_ord"].unique())
    etiquetas = [tabla.loc[tabla["age_ord"] == e, "edad"].iloc[0] for e in edades]
    tope = float(tabla["n"].max()) * 1.32
    # La escala de color se fija con los grupos de al menos 200 encuentros:
    # los decenios jóvenes, con pocas decenas de casos, tienen tasas extremas
    # que estirarían la escala y dejarían iguales a todas las barras grandes.
    fiables = tabla.loc[tabla["n"] >= 200, "tasa"]
    referencia = fiables if len(fiables) >= 2 else tabla["tasa"]
    cmin, cmax = float(referencia.min()), float(referencia.max())
    fig = go.Figure()
    for sexo, signo in [("Mujer", -1), ("Hombre", 1)]:
        sub = tabla[tabla["sexo"] == sexo].set_index("age_ord").reindex(edades)
        # Color calculado aquí y no con coloraxis: Plotly avisaría en consola
        # al derivar el fondo del cuadro emergente de un color numérico.
        posicion = ((sub["tasa"].fillna(cmin) - cmin) / max(cmax - cmin, 1e-9)).clip(0, 1)
        colores = sample_colorscale(ESCALA_AMBAR, posicion.tolist())
        fig.add_trace(go.Bar(
            y=etiquetas, x=signo * sub["n"].fillna(0), orientation="h", name=sexo,
            marker=dict(color=colores, line=dict(color=TINTA_TENUE, width=0.6)),
            text=[pct(t) if pd.notna(t) else "" for t in sub["tasa"]],
            textposition="outside", cliponaxis=False, textfont=dict(size=11, color=TINTA),
            customdata=np.column_stack([sub["n"].fillna(0), sub["tasa"]]),
            hovertemplate=f"<b>{sexo}</b>, %{{y}}<br>%{{customdata[0]:,}} encuentros"
                          "<br>Readmisión: %{customdata[1]:.1f} %<extra></extra>"))
    marcas = np.linspace(0, tope / 1.32, 4)
    fig.update_xaxes(range=[-tope, tope], tickvals=list(-marcas[::-1]) + list(marcas[1:]),
                     ticktext=[miles(abs(v)) for v in list(-marcas[::-1]) + list(marcas[1:])],
                     title_text="Encuentros (mujeres a la izquierda, hombres a la derecha)",
                     showgrid=True, gridcolor=REJILLA)
    fig.add_vline(x=0, line=dict(color=EJE, width=1))
    for texto, x in [("Mujeres", -tope * 0.98), ("Hombres", tope * 0.98)]:
        fig.add_annotation(x=x, y=1.04, yref="paper", text=f"<b>{texto}</b>", showarrow=False,
                           xanchor="left" if x < 0 else "right", font=dict(size=12, color=TINTA))
    # Barra de color de referencia: un punto invisible que solo aporta la escala.
    fig.add_trace(go.Scatter(x=[0], y=[etiquetas[0]], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=0.1, opacity=0, color=[cmin], colorscale=ESCALA_AMBAR,
                                         cmin=cmin, cmax=cmax, showscale=True,
                                         colorbar=dict(title=dict(text="Readmisión (%)", side="right"),
                                                       thickness=12, ticksuffix=" %"))))
    fig.update_layout(barmode="overlay", bargap=0.18)
    fiable = tabla[tabla["n"] >= 200]
    alto = fiable.loc[fiable["tasa"].idxmax()] if len(fiable) else None
    total = tabla.groupby("sexo")["n"].sum()
    mayores = 100 * tabla.loc[tabla["age_ord"].between(5, 8), "n"].sum() / tabla["n"].sum()
    lectura = [f"El {pct(mayores)} de los encuentros son de personas de 50 a 89 años; las mujeres son el "
               f"{pct(100 * total.get('Mujer', 0) / total.sum())} del total."]
    if alto is not None:
        quien = "las mujeres" if alto["sexo"] == "Mujer" else "los hombres"
        lectura.append(f"El grupo con más readmisión (entre los de al menos 200 encuentros) son {quien} "
                       f"de {alto['edad']} años: {pct(alto['tasa'])}.")
    lectura.append("El color de cada barra es su tasa de readmisión: cuanto más oscuro, más reingresos "
                   "(la escala se ajusta a los grupos con al menos 200 encuentros).")
    return tematizar(fig, alto=460, showlegend=False, margin=dict(l=80, r=30, t=40, b=50)), lectura


def fig_dosis(tabla: pd.DataFrame, variable: str, resaltado, prevalencia: float):
    if tabla.empty:
        return _vacio(), []
    x = tabla["etiqueta"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(tabla["ic_alto"]) + list(tabla["ic_bajo"])[::-1],
                             fill="toself", fillcolor=rgba(MEDIANOCHE, 0.12), line=dict(width=0),
                             mode="lines", hoverinfo="skip", name="IC 95 %"))
    fig.add_trace(go.Scatter(
        x=x, y=tabla["tasa"], mode="lines+markers", name="Tasa de readmisión",
        line=dict(color=COLOR_MARCA, width=2.5), marker=dict(size=9, color=COLOR_MARCA, line=dict(color=FONDO, width=2)),
        customdata=np.column_stack([tabla["n"], tabla["ic_bajo"], tabla["ic_alto"]]),
        hovertemplate="%{x}: %{y:.1f} % (IC 95 %: %{customdata[1]:.1f}–%{customdata[2]:.1f})"
                      "<br>n = %{customdata[0]:,}<extra></extra>"))
    fig.add_hline(y=prevalencia, line=dict(color=TINTA_TENUE, dash="dot", width=1),
                  annotation_text=f"media {pct(prevalencia)}", annotation_position="top left",
                  annotation_font=dict(size=11, color=TINTA_TENUE))
    lectura = []
    fila = tabla.iloc[min(int(resaltado or 0), len(tabla) - 1)]
    fig.add_trace(go.Scatter(x=[fila["etiqueta"]], y=[fila["tasa"]], mode="markers", hoverinfo="skip",
                             marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(color=CLASE[1], width=3)),
                             showlegend=False))
    uno_de = 100 / fila["tasa"] if fila["tasa"] > 0 else np.inf
    fig.add_annotation(x=fila["etiqueta"], y=fila["ic_alto"], text=f"<b>1 de cada {num(uno_de, 0)}</b>",
                       showarrow=True, arrowhead=0, ay=-34, font=dict(size=12, color=TINTA))
    lectura.append(f"Con {C.nombre(variable).lower()} = {fila['etiqueta']}, reingresa el {pct(fila['tasa'])} "
                   f"(1 de cada {num(uno_de, 0)}), frente al {pct(prevalencia)} de la media; "
                   f"n = {miles(fila['n'])}.")
    primero, ultimo = tabla.iloc[0], tabla.iloc[-1]
    if ultimo["tasa"] > primero["tasa"]:
        lectura.append(f"La tasa crece de {pct(primero['tasa'])} ({primero['etiqueta']}) a "
                       f"{pct(ultimo['tasa'])} ({ultimo['etiqueta']}): "
                       f"{num(ultimo['tasa'] / max(primero['tasa'], 1e-9), 1)} veces más.")
    else:
        lectura.append(f"La tasa no sube de forma sostenida: {pct(primero['tasa'])} en {primero['etiqueta']} "
                       f"y {pct(ultimo['tasa'])} en {ultimo['etiqueta']}.")
    lectura.append("La banda es el intervalo de confianza de Wilson al 95 %: se ensancha donde hay "
                   "pocos pacientes.")
    fig.update_xaxes(type="category", title_text=C.nombre(variable))
    fig.update_yaxes(title_text="Readmisión en < 30 días (%)", rangemode="tozero")
    return tematizar(con_guia(fig), alto=420, showlegend=False, margin=dict(l=60, r=20, t=40, b=60)), lectura


ORDEN_ESTADOS = ["No", "Steady", "Cambio"]


def fig_farmacos(tabla: pd.DataFrame, prevalencia: float, minimo: int = 30):
    if tabla.empty:
        return _vacio(), []
    comun = tabla[tabla["tipo"] == "estado"]
    orden = comun.drop_duplicates("farmaco").sort_values("prescritos", ascending=False)
    filas = orden["nombre"].tolist()
    estados = [e for e in ORDEN_ESTADOS if e in set(comun["estado"])]
    z, texto, n, fondo_t = [], [], [], []
    for farmaco in orden["farmaco"]:
        sub = comun[comun["farmaco"] == farmaco].set_index("estado")
        fila_z, fila_t, fila_n, fila_f = [], [], [], []
        for e in estados:
            cuenta = int(sub.loc[e, "n"]) if e in sub.index else 0
            if cuenta >= minimo:
                fila_z.append(sub.loc[e, "tasa"])
                fila_t.append(f"{num(sub.loc[e, 'tasa'], 1)} %")
                fila_f.append("")
            else:
                fila_z.append(None)
                fila_t.append("")
                fila_f.append("sin casos" if cuenta == 0 else f"n = {cuenta}")
            fila_n.append(cuenta)
        z.append(fila_z), texto.append(fila_t), n.append(fila_n), fondo_t.append(fila_f)
    cmin, cmax = _divergente_centrado(comun.loc[comun["n"] >= minimo, "tasa"], prevalencia)
    etiquetas_x = ["No prescrito", "Dosis estable", "Dosis cambia"][:len(estados)]
    fig = go.Figure()
    # Capa de fondo: las celdas con pocos casos se ven como celdas neutras
    # con su conteo, no como huecos (un hueco parece un error del gráfico).
    fig.add_trace(go.Heatmap(
        z=[[1] * len(estados) for _ in filas], x=etiquetas_x, y=filas, text=fondo_t, texttemplate="%{text}",
        textfont=dict(size=10, color=TINTA_TENUE), colorscale=[[0, rgba("#8A93B0", 0.16)], [1, rgba("#8A93B0", 0.16)]],
        showscale=False, xgap=3, ygap=3, hoverinfo="skip"))
    fig.add_trace(go.Heatmap(
        z=z, x=etiquetas_x, y=filas, text=texto, texttemplate="%{text}",
        textfont=dict(size=11), colorscale=DIVERGENTE, zmin=cmin, zmax=cmax, zmid=prevalencia,
        xgap=3, ygap=3, customdata=n,
        colorbar=dict(title=dict(text="Readmisión (%)", side="right"), thickness=12, ticksuffix=" %"),
        hovertemplate="<b>%{y}</b> · %{x}<br>Readmisión: %{z:.1f} %<br>n = %{customdata:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    validos = comun[(comun["n"] >= minimo) & (comun["estado"] != "No")]
    lectura = []
    insulina = tabla[(tabla["farmaco"] == "insulin") & (tabla["tipo"] == "estado_original")].set_index("estado")
    sin = comun[(comun["farmaco"] == "insulin") & (comun["estado"] == "No")]
    if {"Up", "Down"} <= set(insulina.index) and len(sin):
        lectura.append(f"Insulina: cuando la dosis cambia durante el ingreso, la readmisión es del "
                       f"{pct(insulina.loc['Up', 'tasa'])} si sube y del {pct(insulina.loc['Down', 'tasa'])} si "
                       f"baja, frente al {pct(sin['tasa'].iloc[0])} sin insulina.")
    if len(validos):
        alto = validos.loc[validos["tasa"].idxmax()]
        lectura.append(f"La combinación con más readmisión (con al menos {minimo} encuentros) es "
                       f"{alto['nombre'].lower()} con «{alto['estado_legible'].lower()}»: {pct(alto['tasa'])}.")
    lectura.append("Un ajuste de dosis señala un paciente menos estable, no un efecto del fármaco: "
                   "es asociación, no causalidad.")
    return tematizar(fig, alto=110 + 30 * len(filas), margin=dict(l=150, r=20, t=50, b=20)), lectura


ORDEN_HBA1C = ["No medida", "Norm", ">7", ">8"]


def fig_hba1c(tabla: pd.DataFrame, prevalencia: float):
    """Tasa con su intervalo, como punto y segmento; la cifra va justo encima
    del extremo superior del intervalo, nunca sobre la línea."""
    if tabla.empty:
        return _vacio(), []
    niveles = [v for v in ORDEN_HBA1C if v in set(tabla["A1Cresult"])]
    posicion = {v: i for i, v in enumerate(niveles)}
    fig = go.Figure()
    for cambio, color, desplazamiento in [("No", COLOR_MARCA, -0.17), ("Ch", COLOR_SEGUNDO, 0.17)]:
        sub = tabla[tabla["change"] == cambio]
        etiqueta = C.nivel("change", cambio)
        x = [posicion[v] + desplazamiento for v in sub["A1Cresult"]]
        fig.add_trace(go.Scatter(
            x=x, y=sub["tasa"], mode="markers", name=etiqueta,
            marker=dict(size=13, color=color, symbol="circle" if cambio == "No" else "diamond",
                        line=dict(color=FONDO, width=2)),
            error_y=dict(type="data", array=sub["ic_alto"] - sub["tasa"],
                         arrayminus=sub["tasa"] - sub["ic_bajo"], color=color, thickness=2, width=7),
            customdata=np.column_stack([sub["n"], sub["ic_bajo"], sub["ic_alto"]]),
            hovertemplate=f"{etiqueta}<br>Readmisión: %{{y:.1f}} % (IC 95 %: %{{customdata[1]:.1f}}–"
                          "%{customdata[2]:.1f})<br>n = %{customdata[0]:,}<extra></extra>"))
        for xi, tasa, alto in zip(x, sub["tasa"], sub["ic_alto"]):
            fig.add_annotation(x=xi, y=alto, text=f"<b>{pct(tasa)}</b>", showarrow=False, yanchor="bottom",
                               yshift=4, font=dict(size=11, color=TINTA))
    # La media se rotula fuera del área de datos, en el margen derecho, para
    # que su etiqueta nunca pise un punto ni su intervalo.
    fig.add_hline(y=prevalencia, line=dict(color=TINTA_TENUE, dash="dot", width=1))
    fig.add_annotation(x=1, xref="paper", y=prevalencia, xanchor="left", xshift=6, showarrow=False,
                       text=f"media<br>{pct(prevalencia)}", align="left", font=dict(size=11, color=TINTA_TENUE))
    # Un gráfico de puntos no necesita empezar en cero (no hay barras cuya
    # longitud se compare): se ajusta a los intervalos para que se lean.
    bajo, alto = float(tabla["ic_bajo"].min()), float(tabla["ic_alto"].max())
    margen = max(0.8, (alto - bajo) * 0.18)
    fig.update_xaxes(title_text="Resultado de la HbA1c durante el ingreso", tickvals=list(range(len(niveles))),
                     ticktext=[C.nivel("A1Cresult", v) for v in niveles], range=[-0.6, len(niveles) - 0.4],
                     showgrid=False)
    fig.update_yaxes(title_text="Readmisión en < 30 días (%)", range=[max(0, bajo - margen), alto + 1.6 * margen],
                     ticksuffix=" %")
    fig.update_layout(legend=dict(title_text="Medicación antidiabética: "))
    medida = tabla.groupby(tabla["A1Cresult"] != "No medida").apply(
        lambda d: 100 * d["k"].sum() / d["n"].sum(), include_groups=False)
    porcentaje_medida = 100 * tabla.loc[tabla["A1Cresult"] != "No medida", "n"].sum() / tabla["n"].sum()
    lectura = [f"Solo al {pct(porcentaje_medida)} de los encuentros se le midió la HbA1c."]
    if True in medida.index and False in medida.index:
        lectura.append(f"Readmisión con la HbA1c medida: {pct(medida[True])}; sin medir: {pct(medida[False])}.")
    lectura.append("El artículo que publicó estos datos (Strack et al., 2014) estudió justamente si medir "
                   "la HbA1c se asocia con menos readmisiones; aquí se ve la misma pregunta con los "
                   "filtros que elijas.")
    return tematizar(fig, alto=420, margin=dict(l=60, r=70, t=50, b=60)), lectura


def fig_comparador(a: pd.Series, b: pd.Series, nombre_a: str, nombre_b: str):
    if a.isna().all() and b.isna().all():
        return _vacio("Ninguno de los dos grupos tiene encuentros."), []
    indicadores = list(a.index)[::-1]
    tope = float(np.nanmax([a.max(), b.max(), 10])) * 1.25
    fig = go.Figure()
    for serie, signo, color, nombre_g in [(a, -1, COLOR_MARCA, nombre_a), (b, 1, COLOR_SEGUNDO, nombre_b)]:
        valores = serie.reindex(indicadores)
        fig.add_trace(go.Bar(
            y=indicadores, x=signo * valores.fillna(0), orientation="h", name=nombre_g,
            marker=dict(color=color, line=dict(color=FONDO, width=1)),
            text=[pct(v, 0) if pd.notna(v) else "–" for v in valores], textposition="outside",
            cliponaxis=False, textfont=dict(size=11, color=TINTA),
            hovertemplate=f"<b>{nombre_g}</b><br>%{{y}}: %{{text}}<extra></extra>"))
    marcas = [v for v in [0, 25, 50, 75, 100] if v <= tope]
    fig.update_xaxes(range=[-tope, tope], tickvals=[-m for m in marcas[::-1]] + marcas[1:],
                     ticktext=[f"{m} %" for m in marcas[::-1]] + [f"{m} %" for m in marcas[1:]],
                     showgrid=True, gridcolor=REJILLA, title_text="Porcentaje de los encuentros del grupo")
    fig.add_vline(x=0, line=dict(color=EJE, width=1))
    for texto, x in [(nombre_a, -tope * 0.98), (nombre_b, tope * 0.98)]:
        fig.add_annotation(x=x, y=1.05, yref="paper", text=f"<b>{texto}</b>", showarrow=False,
                           xanchor="left" if x < 0 else "right", font=dict(size=12, color=TINTA))
    fig.update_layout(barmode="overlay", bargap=0.3)
    diferencias = (b - a).dropna()
    lectura = []
    if len(diferencias):
        clave = diferencias.abs().idxmax()
        grupo_a, grupo_b = (f"el grupo {n.split(':')[0]}" for n in (nombre_a, nombre_b))
        lectura.append(f"La mayor diferencia está en «{clave.lower()}»: {pct(a[clave], 0)} en {grupo_a} "
                       f"frente a {pct(b[clave], 0)} en {grupo_b}.")
        r = "Readmitidos en < 30 días"
        if pd.notna(a.get(r)) and pd.notna(b.get(r)):
            lectura.append(f"Readmisión: {pct(a[r])} en {grupo_a} y {pct(b[r])} en {grupo_b}.")
    return tematizar(fig, alto=470, showlegend=False, margin=dict(l=200, r=30, t=40, b=50)), lectura


def fig_paralelas(tabla: pd.DataFrame, dimensiones: list[str]):
    if tabla.empty:
        return _vacio()
    dims = [dict(values=tabla[d], label=C.nombre(d)) for d in dimensiones]
    dims.append(dict(values=tabla[Y].map(C.CLASES_CORTAS), label="Desenlace",
                     categoryorder="array", categoryarray=[C.CLASES_CORTAS[0], C.CLASES_CORTAS[1]]))
    fig = go.Figure(go.Parcats(
        dimensions=dims, counts=tabla["n"], arrangement="freeform", hoveron="color",
        line=dict(color=tabla[Y], colorscale=[[0, CLASE[0]], [1, CLASE[1]]], cmin=0, cmax=1,
                  shape="hspline"),
        labelfont=dict(size=12, color=TINTA), tickfont=dict(size=11, color=TINTA),
        hovertemplate="%{count:,} encuentros (%{probability:.1%})<extra></extra>",
    ))
    return tematizar(fig, alto=500, margin=dict(l=40, r=60, t=40, b=20))


# =========================================================================== #
# Modelos
# =========================================================================== #
def fig_capacidad(curvas: dict, modelo: str, porcentaje: int):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Al azar",
                             line=dict(color=TINTA_TENUE, dash="dot", width=1), hoverinfo="skip"))
    for otro, curva in curvas.items():
        if otro == modelo:
            continue
        fig.add_trace(go.Scatter(x=[0] + curva["porcentaje"].tolist(), y=[0] + curva["deteccion"].tolist(),
                                 mode="lines", name=C.MODELOS.get(otro, otro),
                                 line=dict(color=COLOR_TENUE, width=1.2),
                                 hovertemplate=f"{C.MODELOS.get(otro, otro)}: %{{x}} % seguidos → "
                                               "%{y:.1f} % detectadas<extra></extra>"))
    curva = curvas[modelo]
    fig.add_trace(go.Scatter(x=[0] + curva["porcentaje"].tolist(), y=[0] + curva["deteccion"].tolist(),
                             mode="lines", name=C.MODELOS.get(modelo, modelo),
                             line=dict(color=COLOR_MARCA, width=3),
                             customdata=[0] + curva["precision"].tolist(),
                             hovertemplate="%{x} % seguidos → %{y:.1f} % de las readmisiones detectadas"
                                           "<br>Precisión: %{customdata:.1f} %<extra></extra>"))
    punto = curva.set_index("porcentaje").loc[int(porcentaje)]
    fig.add_vline(x=porcentaje, line=dict(color=CLASE[1], width=1.5, dash="dash"))
    fig.add_trace(go.Scatter(x=[porcentaje], y=[punto["deteccion"]], mode="markers", hoverinfo="skip",
                             marker=dict(size=13, color=CLASE[1], line=dict(color=FONDO, width=2)),
                             showlegend=False))
    fig.add_annotation(x=porcentaje, y=punto["deteccion"], ax=40, ay=30, showarrow=True, arrowhead=0,
                       text=f"<b>{pct(punto['deteccion'], 0)}</b> detectadas",
                       font=dict(size=12, color=TINTA))
    fig.update_xaxes(title_text="Pacientes que el servicio puede seguir (% de mayor riesgo)", range=[0, 100],
                     ticksuffix=" %")
    fig.update_yaxes(title_text="Readmisiones detectadas (%)", range=[0, 102], ticksuffix=" %")
    return tematizar(con_guia(fig), alto=380, showlegend=False, margin=dict(l=60, r=20, t=20, b=50))


def fig_decision(tabla: pd.DataFrame, modelo: str, umbral_operativo: float | None = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tabla["umbral"], y=tabla["ninguno"], mode="lines", name="No intervenir a nadie",
                             line=dict(color=TINTA_TENUE, width=1.5),
                             hovertemplate="Umbral %{x:.2f}: beneficio neto 0<extra></extra>"))
    fig.add_trace(go.Scatter(x=tabla["umbral"], y=tabla["todos"], mode="lines", name="Intervenir a todos",
                             line=dict(color=TINTA_TENUE, dash="dot", width=1.5),
                             hovertemplate="Umbral %{x:.2f}: %{y:.4f}<extra>Intervenir a todos</extra>"))
    fig.add_trace(go.Scatter(x=tabla["umbral"], y=tabla["modelo"], mode="lines",
                             name=C.MODELOS.get(modelo, modelo), line=dict(color=COLOR_MARCA, width=3),
                             hovertemplate="Umbral %{x:.2f}: %{y:.4f}<extra>Modelo</extra>"))
    if umbral_operativo is not None:
        fig.add_vline(x=umbral_operativo, line=dict(color=CLASE[1], dash="dash", width=1.2),
                      annotation_text="umbral operativo", annotation_position="top right",
                      annotation_font=dict(size=11, color=TINTA_TENUE))
    tope = max(float(tabla["modelo"].max()), float(tabla["todos"].max())) * 1.15
    fig.update_xaxes(title_text="Umbral de probabilidad para intervenir", range=[0.02, 0.5])
    fig.update_yaxes(title_text="Beneficio neto (readmisiones netas por paciente)",
                     range=[-0.01, max(tope, 0.02)], zeroline=False)
    gana = tabla[(tabla["modelo"] > tabla["todos"]) & (tabla["modelo"] > 0)]
    lectura = []
    if len(gana):
        lectura.append(f"El modelo aporta beneficio neto frente a intervenir a todos o a nadie entre los umbrales "
                       f"{num(gana['umbral'].min(), 2)} y {num(gana['umbral'].max(), 2)}.")
    else:
        lectura.append("En ningún umbral el modelo supera a las estrategias de intervenir a todos o a nadie.")
    lectura.append("Cómo leerla: un umbral de 0,20 significa aceptar 4 intervenciones innecesarias por cada "
                   "readmisión detectada (0,20 / 0,80). La curva más alta es la mejor decisión para ese umbral.")
    return tematizar(fig, alto=360, margin=dict(l=70, r=20, t=40, b=50)), lectura


def fig_termometro(p: float, umbral: float, decision: bool):
    tope = max(0.4, min(1.0, float(np.ceil(max(p, umbral) * 1.4 * 10) / 10)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * p,
        number=dict(suffix=" %", valueformat=".1f", font=dict(size=40, color=TINTA)),
        gauge=dict(
            shape="angular",
            axis=dict(range=[0, 100 * tope], ticksuffix=" %", tickfont=dict(size=11, color=TINTA_TENUE)),
            bar=dict(color=CLASE[1] if decision else CLASE[0], thickness=0.32),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, 100 * umbral], color=rgba(CLASE[0], 0.14)),
                   dict(range=[100 * umbral, 100 * tope], color=rgba(CLASE[1], 0.18))],
            threshold=dict(line=dict(color=TINTA, width=3), thickness=0.85, value=100 * umbral)),
    ))
    fig.add_annotation(x=0.5, y=-0.08, xref="paper", yref="paper", showarrow=False,
                       text=f"Línea negra: umbral operativo ({pct(100 * umbral)})",
                       font=dict(size=11, color=TINTA_TENUE))
    return tematizar(fig, alto=250, margin=dict(l=30, r=30, t=20, b=30))


def valor_legible(variable: str, valor) -> str:
    """Valor de una variable tal como lo leería un clínico: «70–79 años»,
    «Dosis baja», «Otro hospital» en lugar de los códigos del conjunto."""
    if variable == "age_ord":
        return f"{A.edad_legible(valor)} años"
    if variable in D.roles()["binarias"]:
        return "Sí" if int(valor) == 1 else "No"
    if variable in D.roles()["categoricas"]:
        return C.nivel(variable, valor)
    if isinstance(valor, float) and valor.is_integer():
        return str(int(valor))
    return str(valor)


def fig_shap_cascada(contribuciones: pd.Series, caso: pd.DataFrame, base: float, prediccion: float,
                     etiquetar=None, top: int = 10):
    """Cascada: de la tasa base a la probabilidad del caso, variable a variable."""
    orden = contribuciones.abs().sort_values(ascending=False)
    principales = contribuciones[orden.index[:top]]
    resto = float(contribuciones[orden.index[top:]].sum())
    etiquetas = ["Tasa base del modelo"]
    etiquetar = etiquetar or valor_legible
    etiquetas += [f"{C.nombre(v)} = {etiquetar(v, caso[v].iloc[0])}" for v in principales.index]
    valores = [100 * base] + [100 * v for v in principales.values]
    medidas = ["absolute"] + ["relative"] * len(principales)
    if len(orden) > top:
        etiquetas.append(f"Otras {len(orden) - top} variables")
        valores.append(100 * resto)
        medidas.append("relative")
    etiquetas.append("Probabilidad de este paciente")
    valores.append(100 * prediccion)
    medidas.append("total")
    textos = [pct(valores[0])] + [f"{num(v, 1, signo=True)} pp" for v in valores[1:-1]] + [pct(valores[-1])]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=etiquetas, x=valores, measure=medidas, text=textos, textposition="outside",
        cliponaxis=False, connector=dict(line=dict(color=EJE, width=1)),
        increasing=dict(marker=dict(color=CLASE[1])), decreasing=dict(marker=dict(color=CLASE[0])),
        totals=dict(marker=dict(color=COLOR_MARCA)),
        hovertemplate="%{y}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Probabilidad de readmisión (%)", ticksuffix=" %", showgrid=True,
                     gridcolor=REJILLA, rangemode="tozero")
    return tematizar(fig, alto=90 + 30 * len(etiquetas), showlegend=False,
                     margin=dict(l=270, r=70, t=10, b=50))


def fig_que_pasa_si(valores, probabilidades, actual, variable: str, umbral: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=valores, y=100 * np.asarray(probabilidades), mode="lines+markers",
                             line=dict(color=COLOR_MARCA, width=2.5),
                             marker=dict(size=7, color=COLOR_MARCA, line=dict(color=FONDO, width=1.5)),
                             hovertemplate=f"{C.nombre(variable)} = %{{x}}: %{{y:.1f}} %<extra></extra>"))
    if actual in list(valores):
        i = list(valores).index(actual)
        fig.add_trace(go.Scatter(x=[actual], y=[100 * probabilidades[i]], mode="markers+text",
                                 text=["este paciente"], textposition="top center",
                                 textfont=dict(size=11, color=TINTA), hoverinfo="skip",
                                 marker=dict(size=14, color=CLASE[1], line=dict(color=FONDO, width=2))))
    fig.add_hline(y=100 * umbral, line=dict(color=TINTA_TENUE, dash="dot", width=1),
                  annotation_text="umbral operativo", annotation_position="top left",
                  annotation_font=dict(size=11, color=TINTA_TENUE))
    fig.update_xaxes(title_text=C.nombre(variable))
    fig.update_yaxes(title_text="Probabilidad de readmisión (%)", ticksuffix=" %", rangemode="tozero")
    return tematizar(con_guia(fig), alto=320, showlegend=False, margin=dict(l=60, r=20, t=30, b=50))


def fig_consenso(tabla: pd.DataFrame, seleccionado: str):
    """Probabilidad del mismo paciente según los 7 modelos.

    El color dice si ese modelo, con su propio umbral operativo, lo
    priorizaría; la raya vertical es ese umbral.
    """
    tabla = tabla.sort_values("p")
    nombres = [C.MODELOS.get(m, m) for m in tabla["modelo"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=100 * tabla["umbral"], y=nombres, mode="markers", name="Umbral del modelo",
                             marker=dict(symbol="line-ns", size=18, line=dict(color=TINTA_TENUE, width=2)),
                             hovertemplate="Umbral de %{y}: %{x:.1f} %<extra></extra>"))
    for clase, nombre_c in [(0, "No lo priorizaría"), (1, "Lo priorizaría")]:
        sub = tabla[(tabla["p"] >= tabla["umbral"]).astype(int) == clase]
        fig.add_trace(go.Scatter(
            x=100 * sub["p"], y=[C.MODELOS.get(m, m) for m in sub["modelo"]], mode="markers+text",
            name=nombre_c, text=[pct(100 * v) for v in sub["p"]], textposition="middle right",
            textfont=dict(size=11, color=TINTA),
            marker=dict(size=[18 if m == seleccionado else 12 for m in sub["modelo"]], color=CLASE[clase],
                        symbol="diamond" if clase else "circle", line=dict(color=FONDO, width=2)),
            hovertemplate="%{y}: %{x:.1f} %<extra>" + nombre_c + "</extra>"))
    tope = float(max(tabla["p"].max(), tabla["umbral"].max())) * 100 * 1.3
    fig.update_xaxes(title_text="Probabilidad de readmisión (%)", range=[0, max(tope, 10)], ticksuffix=" %",
                     showgrid=True, gridcolor=REJILLA)
    return tematizar(fig, alto=330, margin=dict(l=150, r=30, t=40, b=50))


def fig_rangos(rangos: pd.DataFrame, original: str | None = None):
    """Rango de cada modelo en cada fold: ¿el orden se mantiene?

    Se destacan los dos mejores por rango medio y, si se indica, el modelo
    original (en tinta, como en el diagrama de diferencia crítica).
    """
    medios = rangos.mean().sort_values()
    destacados = list(medios.index[:2])
    fig = go.Figure()
    for modelo in medios.index[::-1]:
        destacado = modelo in destacados or modelo == original
        color = (TINTA if modelo == original else COLOR_MARCA if modelo == destacados[0]
                 else AZUR if destacado else COLOR_TENUE)
        fig.add_trace(go.Scatter(
            x=rangos.index, y=rangos[modelo], mode="lines+markers", name=C.nombre_modelo(modelo),
            # Raya discontinua para el original: en modo oscuro su tinta y el
            # color del mejor se aclaran casi igual, y el trazo los separa.
            line=dict(color=color, width=3 if destacado else 1.5,
                      dash="dash" if modelo == original else "solid"),
            marker=dict(size=10 if destacado else 7, color=color, line=dict(color=FONDO, width=2)),
            hovertemplate=f"{C.nombre_modelo(modelo)} · %{{x}}: puesto %{{y:.0f}}<extra></extra>"))
        fig.add_annotation(x=rangos.index[-1], y=rangos[modelo].iloc[-1], xanchor="left", xshift=12,
                           text=C.nombre_modelo(modelo), showarrow=False,
                           font=dict(size=11, color=TINTA if destacado else TINTA_TENUE))
    fig.update_yaxes(autorange="reversed", dtick=1, title_text="Puesto en AUC-PR (1 = mejor)")
    fig.update_xaxes(title_text="Fold externo de la validación anidada")
    return tematizar(fig, alto=380, showlegend=False, margin=dict(l=60, r=150, t=20, b=50))


def fig_equidad(tabla: pd.DataFrame, global_: dict, atributo: str):
    paneles = [("auc_roc", "AUC-ROC", False), ("deteccion", "Readmisiones detectadas (%)", True),
               ("marcados", "Pacientes priorizados (%)", True)]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=[p[1] for p in paneles])
    grupos = tabla["grupo"].tolist()
    for j, (columna, titulo, es_pct) in enumerate(paneles, start=1):
        colores = [COLOR_MARCA if f else COLOR_TENUE for f in tabla["fiable"]]
        fig.add_trace(go.Scatter(
            x=tabla[columna], y=grupos, mode="markers", marker=dict(size=12, color=colores,
                                                                   line=dict(color=FONDO, width=2)),
            customdata=np.column_stack([tabla["n"], tabla["readmitidos"]]),
            hovertemplate="%{y}: %{x:.3f}" if not es_pct else "%{y}: %{x:.1f} %",
            showlegend=False), row=1, col=j)
        fig.data[-1].hovertemplate += "<br>n = %{customdata[0]:,} · readmitidos = %{customdata[1]:,}<extra></extra>"
        fig.add_vline(x=global_[columna], line=dict(color=TINTA_TENUE, dash="dot", width=1), row=1, col=j)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showgrid=True, gridcolor=REJILLA)
    fig.update_annotations(font=dict(size=12, color=TINTA))
    fiables = tabla[tabla["fiable"]]
    lectura = []
    if len(fiables) >= 2:
        alto, bajo = fiables.loc[fiables["deteccion"].idxmax()], fiables.loc[fiables["deteccion"].idxmin()]
        lectura.append(f"Con el umbral operativo, el modelo detecta el {pct(alto['deteccion'])} de las "
                       f"readmisiones en «{alto['grupo']}» y el {pct(bajo['deteccion'])} en «{bajo['grupo']}».")
        rango_auc = fiables["auc_roc"].max() - fiables["auc_roc"].min()
        lectura.append(f"El AUC-ROC varía {num(rango_auc, 3)} entre grupos: "
                       + ("una diferencia pequeña frente a la incertidumbre de la prueba."
                          if rango_auc < 0.03 else "conviene vigilarla antes de usar el modelo en esos grupos."))
    lectura.append("Gris: grupos con menos de 30 readmitidos en prueba, cuyas métricas son inestables. "
                   "La línea punteada es el valor en toda la prueba.")
    return tematizar(fig, alto=110 + 42 * len(grupos), margin=dict(l=120, r=20, t=50, b=40)), lectura


# =========================================================================== #
# Sustitutos del Sankey, del sunburst y de las categorías paralelas
# =========================================================================== #
def _barras_tasa(tabla: pd.DataFrame, etiquetas, prevalencia: float, seleccionado=None,
                 resaltar_por_media: bool = True, atenuado: float = 0.3):
    """Barras horizontales de tasa con su IC: el formato más directo para
    comparar tasas entre grupos. Ámbar si el grupo supera la media, azul si
    no; el seleccionado va opaco y los demás atenuados."""
    colores = []
    for nivel, tasa in zip(tabla["nivel"], tabla["tasa"]):
        base = CLASE[1] if (resaltar_por_media and tasa > prevalencia) else CLASE[0]
        colores.append(base if seleccionado in (None, A.CUALQUIERA, nivel) else rgba(base, atenuado))
    fig = go.Figure(go.Bar(
        x=tabla["tasa"], y=etiquetas, orientation="h", marker=dict(color=colores, line=dict(width=0)),
        error_x=dict(type="data", array=tabla["ic_alto"] - tabla["tasa"], arrayminus=tabla["tasa"] - tabla["ic_bajo"],
                     color=TINTA_TENUE, thickness=1.2, width=3),
        text=[pct(t) for t in tabla["tasa"]], textposition="outside", cliponaxis=False,
        textfont=dict(size=11, color=TINTA),
        customdata=np.column_stack([tabla["n"], tabla["ic_bajo"], tabla["ic_alto"]]),
        hovertemplate="<b>%{y}</b><br>Readmisión: %{x:.1f} %<br>IC 95 %: %{customdata[1]:.1f}–%{customdata[2]:.1f} %"
                      "<br>Encuentros: %{customdata[0]:,}<extra></extra>"))
    fig.add_vline(x=prevalencia, line=dict(color=TINTA_TENUE, dash="dot", width=1))
    tope = float(np.nanmax(tabla["ic_alto"])) * 1.25 if len(tabla) else 20
    fig.update_xaxes(range=[0, max(tope, prevalencia * 1.4)], ticksuffix=" %", showgrid=True, gridcolor=REJILLA)
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_etapa(tabla: pd.DataFrame, prevalencia: float, seleccionado=None):
    if tabla.empty:
        return _vacio()
    fig = _barras_tasa(tabla, list(tabla["nivel"]), prevalencia, seleccionado)
    # Sin zoom ni arrastre: en estas barras el clic sirve para elegir.
    return tematizar(fig, alto=70 + 34 * len(tabla), showlegend=False, bargap=0.35, dragmode=False,
                     margin=dict(l=10, r=40, t=10, b=30))


def fig_ranking_diagnosticos(tabla: pd.DataFrame, prevalencia: float, seleccionado=None, titulo_x=None):
    if tabla.empty:
        return _vacio("No hay diagnósticos con suficientes encuentros.")
    # En el ranking hay que poder comparar todas las barras: el resto se
    # atenúa menos que en el recorrido, donde lo importante es la elección.
    fig = _barras_tasa(tabla, list(tabla["nivel"]), prevalencia, seleccionado, atenuado=0.6)
    fig.add_annotation(x=prevalencia, y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"media {pct(prevalencia)}", font=dict(size=11, color=TINTA_TENUE))
    fig.update_xaxes(title_text=titulo_x or "Readmisión en < 30 días (%)")
    return tematizar(fig, alto=90 + 28 * len(tabla), showlegend=False, bargap=0.3, dragmode=False,
                     margin=dict(l=10, r=50, t=30, b=50))


def fig_matriz_riesgo(tasa: pd.DataFrame, n: pd.DataFrame, fila: str, columna: str, prevalencia: float,
                      minimo: int = 30):
    if tasa.empty:
        return _vacio(), []
    filas, columnas = [str(i) for i in tasa.index], [str(c) for c in tasa.columns]
    z = tasa.where(n >= minimo).to_numpy(dtype=float)
    z_lista = [[None if np.isnan(v) else float(v) for v in fila_z] for fila_z in z]
    texto = [[f"{num(v, 1)} %" if v is not None else "" for v in fila_z] for fila_z in z_lista]
    fondo = [[("" if n.iat[i, j] >= minimo else ("sin casos" if n.iat[i, j] == 0 else f"n = {int(n.iat[i, j])}"))
              for j in range(len(columnas))] for i in range(len(filas))]
    validos = tasa.where(n >= minimo).stack()
    cmin, cmax = _divergente_centrado(validos, prevalencia)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=[[1] * len(columnas) for _ in filas], x=columnas, y=filas, text=fondo,
                             texttemplate="%{text}", textfont=dict(size=10, color=TINTA_TENUE),
                             colorscale=[[0, rgba("#8A93B0", 0.16)], [1, rgba("#8A93B0", 0.16)]],
                             showscale=False, xgap=4, ygap=4, hoverinfo="skip"))
    fig.add_trace(go.Heatmap(
        z=z_lista, x=columnas, y=filas, text=texto, texttemplate="%{text}", textfont=dict(size=12),
        colorscale=DIVERGENTE, zmin=cmin, zmax=cmax, zmid=prevalencia, xgap=4, ygap=4, customdata=n.to_numpy(),
        colorbar=dict(title=dict(text="Readmisión (%)", side="right"), thickness=12, ticksuffix=" %"),
        hovertemplate=f"<b>{C.nombre(fila)}: %{{y}}</b><br><b>{C.nombre(columna)}: %{{x}}</b>"
                      "<br>Readmisión: %{z:.1f} %<br>Encuentros: %{customdata:,}<extra></extra>"))
    fig.update_xaxes(title_text=C.nombre(columna), side="bottom", showgrid=False, type="category")
    fig.update_yaxes(title_text=C.nombre(fila), autorange="reversed", showgrid=False, type="category")
    lectura = []
    if len(validos):
        (f_alta, c_alta), (f_baja, c_baja) = validos.idxmax(), validos.idxmin()
        lectura.append(f"La combinación con más readmisión es {C.nombre(fila).lower()} «{f_alta}» con "
                       f"{C.nombre(columna).lower()} «{c_alta}»: {pct(validos.max())} "
                       f"(n = {miles(n.loc[f_alta, c_alta])}).")
        lectura.append(f"La de menos es «{f_baja}» con «{c_baja}»: {pct(validos.min())}. "
                       f"La media es {pct(prevalencia)}.")
    lectura.append(f"Las celdas con menos de {minimo} encuentros se muestran en gris con su conteo: su tasa "
                   "no sería fiable.")
    return tematizar(fig, alto=max(300, 90 + 52 * len(filas)), margin=dict(l=150, r=20, t=20, b=70)), lectura
