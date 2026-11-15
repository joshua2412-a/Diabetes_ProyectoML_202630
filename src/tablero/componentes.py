"""
Componentes reutilizables de interfaz.

Cada página describe su contenido, no su maquetación. Las piezas que reciben
texto lo pasan por ``con_formulas``: basta con escribir $...$ en la cadena
para que MathJax componga la fórmula.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html

from .formato import num, pct
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
def encabezado(ceja: str, titulo: str, texto: str, ilustracion: str | None = None) -> html.Div:
    """«Antes: después» se compone en dos pesos (fino y negrita).

    ``ilustracion`` es el nombre de un SVG de assets/ que se coloca a la
    derecha de la banda (solo en pantallas anchas).
    """
    if ": " in titulo:
        antes, despues = titulo.split(": ", 1)
        partes = [html.Span(antes + ":", className="titulo-ligero"), html.Br(), despues]
    else:
        partes = [titulo]
    texto_banda = html.Div([
        html.Div(ceja, className="banda-ceja"),
        html.H1(partes, className="banda-titulo"),
        html.P(texto, className="banda-texto"),
    ])
    if ilustracion is None:
        return html.Div(texto_banda, className="banda-pagina")
    return html.Div(html.Div([
        texto_banda,
        html.Img(src=f"/assets/{ilustracion}", className="banda-ilustracion", alt=""),
    ], className="banda-con-ilustracion"), className="banda-pagina")


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
    if alto:  # nunca style=None: dcc.Graph lee style.height y fallaría
        extra["style"] = {"height": f"{int(alto)}px"}
    grafico = dcc.Graph(figure=figura, config={"displaylogo": False, "responsive": True}, **extra)
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


# ---- Matriz de personas (comunicación de riesgo) ------------------------- #
def matriz_personas(grupos: Sequence[dict], descripcion: str, columnas: int = 10) -> html.Div:
    """Matriz de iconos de persona: «de cada 100 pacientes, 11 vuelven».

    Es el formato que recomienda la literatura de comunicación de riesgo para
    público no experto: una frecuencia natural se entiende mejor que un
    porcentaje. Cada grupo es un dict con ``n`` (iconos), ``clase`` (CSS: la
    clase da el color), ``etiqueta`` (leyenda) y, opcionalmente,
    ``relleno`` (icono lleno o de contorno). La forma del icono y la
    leyenda repiten lo que dice el color, y ``descripcion`` es el texto
    alternativo de la figura entera.
    """
    iconos, leyenda = [], []
    for g in grupos:
        estilo = f"persona {g['clase']}" + (" relleno" if g.get("relleno", True) else "")
        iconos += [html.Span(g.get("icono", "person"), className=f"material-symbols-rounded {estilo}",
                             **{"aria-hidden": "true"}) for _ in range(int(g["n"]))]
        if g.get("etiqueta"):
            leyenda.append(html.Span([
                html.Span(g.get("icono", "person"), className=f"material-symbols-rounded {estilo}",
                          **{"aria-hidden": "true"}),
                html.Strong(f"{int(g['n'])} "), g["etiqueta"]], className="persona-leyenda"))
    return html.Div([
        html.Div(iconos, className="matriz-personas", role="img", **{"aria-label": descripcion},
                 style={"gridTemplateColumns": f"repeat({columnas}, 1fr)"}),
        html.Div(leyenda, className="personas-leyenda"),
    ])


def semaforo(estado: str, titulo: str, texto, icono_nombre: str) -> html.Div:
    """Estado con icono, título y texto. ``estado``: bien, aviso, critico o neutro.

    El color nunca va solo: el icono y el título dicen lo mismo.
    """
    return html.Div([
        html.Span(icono_nombre, className="material-symbols-rounded semaforo-icono", **{"aria-hidden": "true"}),
        html.Div([html.Div(titulo, className="semaforo-titulo"),
                  html.Div(con_formulas(texto), className="semaforo-texto")]),
    ], className=f"semaforo semaforo-{estado}")


# ---- Glucómetro de riesgo (predicción en vivo) --------------------------- #
def glucometro(p: float, umbral: float, decision: bool, percentil: float | None) -> html.Div:
    """La probabilidad predicha, leída en un glucómetro.

    Es el aparato que cualquier persona con diabetes reconoce: la pantalla
    muestra la cifra y, debajo, una escala con el umbral operativo y una
    marca que se desliza hasta el valor del paciente. No añade información
    nueva: la presenta en un objeto familiar. Los colores son los de las
    clases (azul: seguimiento habitual; ámbar: priorizar).
    """
    tope = max(0.4, min(1.0, float(np.ceil(max(p, umbral) * 1.4 * 10) / 10)))
    posicion, corte = 100 * min(p, tope) / tope, 100 * umbral / tope
    estado = "Priorizar seguimiento" if decision else "Seguimiento habitual"
    clase = "gluco-alto" if decision else "gluco-bajo"
    return html.Div([
        html.Div(html.Span(className="gluco-gota"), className="gluco-tira", **{"aria-hidden": "true"}),
        html.Div([
            html.Div([
                html.Div("Riesgo de readmisión · 30 días", className="gluco-etiqueta"),
                html.Div(pct(100 * p), className="gluco-numero"),
                html.Div([html.Span(className="gluco-led"), estado], className="gluco-estado"),
            ], className="gluco-pantalla"),
            html.Div([
                html.Div([
                    html.Div(className="gluco-umbral", style={"left": f"{corte:.1f}%"},
                             title=f"Umbral operativo: {pct(100 * umbral)}"),
                    html.Div(className="gluco-marcador", style={"left": f"{posicion:.1f}%"}),
                ], className="gluco-pista"),
                html.Div([html.Span("0 %"), html.Span(f"umbral {pct(100 * umbral)}", className="gluco-txt-umbral",
                                                       style={"left": f"{corte:.1f}%"}),
                          html.Span(pct(100 * tope, 0))], className="gluco-ejes"),
            ], className="gluco-escala"),
            html.Div([html.Span(className="gluco-boton"), html.Span(className="gluco-boton")],
                     className="gluco-botones", **{"aria-hidden": "true"}),
        ], className="gluco-cuerpo"),
        html.Div(f"Percentil {num(percentil, 0)} de riesgo entre los pacientes de prueba"
                 if percentil is not None else "", className="gluco-pie"),
    ], className=f"glucometro {clase}", role="img",
        **{"aria-label": f"Probabilidad de readmisión en menos de 30 días: {pct(100 * p)}. "
                         f"Umbral operativo {pct(100 * umbral)}. Decisión: {estado.lower()}."})
