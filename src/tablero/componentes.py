"""
Componentes reutilizables de interfaz.

Cada página describe su contenido, no su maquetación. Las piezas que reciben
texto lo pasan por ``con_formulas``: basta con escribir $...$ en la cadena
para que MathJax componga la fórmula.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import dash_bootstrap_components as dbc
from dash import dcc, html

from .tema import COLOR_MARCA, TINTA_TENUE


# ---- Texto y fórmulas ---------------------------------------------------- #
def tex(texto: str, clase: str = "") -> dcc.Markdown:
    return dcc.Markdown(texto, mathjax=True, className=f"tex {clase}".strip())


def con_formulas(contenido):
    if isinstance(contenido, str) and "$" in contenido:
        return tex(contenido)
    if isinstance(contenido, list):
        return [con_formulas(parte) for parte in contenido]
    return contenido


def icono(nombre: str, clase: str = "") -> html.Span:
    """Icono de Material Symbols por su nombre. Nunca emojis ni dibujos a mano."""
    return html.Span(nombre, className=f"material-symbols-rounded {clase}".strip(),
                     **{"aria-hidden": "true"})


def parrafo(texto) -> html.Div:
    if isinstance(texto, str) and "$" in texto:
        return tex(texto, "cuerpo")
    return html.P(texto, className="cuerpo")


def lista(items: Iterable) -> html.Ul:
    return html.Ul([html.Li(con_formulas(i)) for i in items], className="lista")


def subtitulo(texto: str) -> html.H3:
    return html.H3(texto, className="subtitulo-seccion")


# ---- Cabecera de página (banda azul noche del estilo B+) ----------------- #
def encabezado(ceja: str, titulo: str, texto: str) -> html.Div:
    """«Antes: después» se compone en dos pesos (fino y negrita)."""
    if ": " in titulo:
        antes, despues = titulo.split(": ", 1)
        partes = [html.Span(antes + ":", className="titulo-ligero"), html.Br(), despues]
    else:
        partes = [titulo]
    return html.Div([
        html.Div(ceja, className="banda-ceja"),
        html.H1(partes, className="banda-titulo"),
        html.P(texto, className="banda-texto"),
    ], className="banda-pagina")


def subpestanas(pestanas: Sequence[tuple], id_: str) -> dbc.Tabs:
    """Subpestañas de una página: lista de (id, etiqueta, contenido)."""
    return dbc.Tabs(
        [dbc.Tab(contenido, label=etiqueta, tab_id=tab_id,
                 label_class_name="subpestana", active_label_class_name="subpestana-activa")
         for tab_id, etiqueta, contenido in pestanas],
        id=id_, active_tab=pestanas[0][0], className="subpestanas")


# ---- Tarjetas ------------------------------------------------------------ #
def tarjeta(contenido, titulo: str | None = None, nota: str | None = None,
            tono: str = "", clase: str = "") -> dbc.Card:
    """Tarjeta genérica. ``tono`` = "acento" (interpreta) o "panel" (agrupa)."""
    cuerpo = [html.H4(titulo, className="tarjeta-titulo")] if titulo else []
    cuerpo.append(html.Div(contenido))
    if nota:
        cuerpo.append(html.Div(con_formulas(nota), className="tarjeta-nota"))
    clases = " ".join(c for c in ("tarjeta", f"tono-{tono}" if tono else "", clase) if c)
    return dbc.Card(dbc.CardBody(cuerpo), className=clases)


def kpi(valor: str, etiqueta: str, detalle: str = "", color: str = COLOR_MARCA) -> dbc.Card:
    """Indicador. La franja es neutra salvo que la cifra hable de una clase."""
    cuerpo = [html.Div(etiqueta, className="kpi-etiqueta"),
              html.Div(valor, className="kpi-valor")]
    if detalle:
        cuerpo.append(html.Div(con_formulas(detalle), className="kpi-detalle"))
    return dbc.Card([html.Div(className="kpi-franja", style={"background": color}),
                     dbc.CardBody(cuerpo)], className="tarjeta kpi")


def fila_kpi(indicadores: Sequence[dict], md: int = 3) -> dbc.Row:
    return dbc.Row([dbc.Col(kpi(**k), md=md, sm=6, xs=12, className="mb-3")
                    for k in indicadores], className="g-3")


def aviso(texto, titulo: str | None = None, tipo: str = "info") -> html.Div:
    """Nota destacada. ``tipo``: info, alcance o pendiente."""
    hijos = [html.Div(titulo, className="aviso-titulo")] if titulo else []
    hijos.append(html.Div(con_formulas(texto), className="aviso-texto"))
    return html.Div(hijos, className=f"aviso aviso-{tipo}")


def lectura(lineas: Iterable, titulo: str = "Lectura") -> dbc.Card:
    """El mensaje del gráfico, junto a él: convierte una figura en un hallazgo."""
    return tarjeta(html.Ul([html.Li(con_formulas(l)) for l in lineas],
                           className="lectura-lista"),
                   titulo=titulo, tono="acento", clase="lectura")


def tarjeta_grafico(figura, titulo: str, nota: str = "", id_: str | None = None,
                    alto: int | None = None) -> dbc.Card:
    """Gráfico con título y nota. La altura de la figura pasa al contenedor."""
    alto = alto or getattr(getattr(figura, "layout", None), "height", None)
    extra = {"id": id_} if id_ else {}
    grafico = dcc.Graph(figure=figura, config={"displaylogo": False, "responsive": True},
                        style={"height": f"{int(alto)}px"} if alto else None, **extra)
    cuerpo = [html.H4(titulo, className="tarjeta-titulo"), grafico]
    if nota:
        cuerpo.append(html.Div(con_formulas(nota), className="tarjeta-nota",
                               style={"color": TINTA_TENUE}))
    return dbc.Card(dbc.CardBody(cuerpo), className="tarjeta")


def tabla(encabezados: Sequence[str], filas: Sequence[Sequence]) -> dbc.Table:
    cabeza = html.Thead(html.Tr([html.Th(h) for h in encabezados]))
    cuerpo = html.Tbody([html.Tr([html.Td(con_formulas(v), className="celda-fuerte" if i == 0 else "")
                                  for i, v in enumerate(f)]) for f in filas])
    return dbc.Table([cabeza, cuerpo], bordered=False, hover=True, responsive=True,
                     className="tabla")


def pendiente(titulo: str, elementos: Iterable[str]) -> dbc.Card:
    """Lo que el Entregable 3 exige y todavía no existe, dicho explícitamente."""
    return tarjeta([
        html.Div([icono("pending", "me-2"), "Pendiente"], className="etiqueta-pendiente"),
        html.Ul([html.Li(e) for e in elementos], className="lista"),
    ], titulo=titulo, clase="tarjeta-pendiente")


def falta_artefacto(que: str) -> html.Div:
    return aviso(
        f"Falta {que}. Ejecuta `python experiments/entrenar_artefactos.py` desde la "
        "raíz del repositorio: el dashboard no entrena modelos, solo lee artifacts/.",
        "Artefactos sin generar", "pendiente")
