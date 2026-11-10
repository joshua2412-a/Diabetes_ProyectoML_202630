"""
Página 2 · EDA: el análisis exploratorio sobre el conjunto de entrenamiento.

Contenido mínimo de la sección 7.3.2 de la guía: filtros globales que
actualizan todos los gráficos, distribución del objetivo, univariado y un
explorador bivariado con ejes y color elegibles, correlación, faltantes y
atípicos, y una lectura junto a cada gráfico.

Cada gráfico se calcula solo cuando su subpestaña está activa: así no se
dibuja en un panel oculto (Plotly lo dejaría sin ancho) y un cambio de
filtro solo recalcula lo que se ve.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html, no_update

from src.tablero import config as C
from src.tablero import datos as D
from src.tablero import graficos as G
from src.tablero.componentes import (aviso, encabezado, falta_artefacto, lectura,
                                     subpestanas, tarjeta, tarjeta_grafico)
from src.tablero.formato import miles, pct

dash.register_page(__name__, path="/eda", name="EDA", order=1,
                   title="EDA · Readmisión temprana")

FILTROS = [Input("f-edad", "value"), Input("f-sexo", "value"), Input("f-raza", "value"),
           Input("f-admision", "value"), Input("f-dias", "value")]
SUBPESTANAS = "subpestanas-eda"


def _filtros():
    o = D.opciones_filtro()
    desplegable = lambda id_, opciones, texto: dcc.Dropdown(
        id=id_, options=opciones, multi=True, placeholder=texto, className="filtro")
    return tarjeta([
        dbc.Row([
            dbc.Col([html.Label("Edad", htmlFor="f-edad"),
                     desplegable("f-edad", o["edad"], "Todas")], md=3),
            dbc.Col([html.Label("Sexo", htmlFor="f-sexo"),
                     desplegable("f-sexo", o["sexo"], "Todos")], md=2),
            dbc.Col([html.Label("Grupo racial", htmlFor="f-raza"),
                     desplegable("f-raza", o["raza"], "Todos")], md=2),
            dbc.Col([html.Label("Tipo de admisión", htmlFor="f-admision"),
                     desplegable("f-admision", o["admision"], "Todos")], md=2),
            dbc.Col([html.Label("Días de hospitalización", htmlFor="f-dias"),
                     dcc.RangeSlider(id="f-dias", min=o["dias"][0], max=o["dias"][1], step=1,
                                     value=list(o["dias"]), allowCross=False,
                                     marks={d: str(d) for d in range(o["dias"][0], o["dias"][1] + 1, 3)},
                                     tooltip={"placement": "bottom"})], md=3),
        ], className="g-3"),
        html.Div(id="eda-n", className="filtros-n", **{"aria-live": "polite"}),
    ], titulo="Filtros (afectan a todos los gráficos de la página)",
        nota="El conjunto no tiene fechas, así que no hay filtro de periodo. Los filtros se "
             "aplican al conjunto de entrenamiento.", clase="tarjeta-filtros")


def _bloque(id_grafico, id_lectura, titulo, nota, controles=None):
    """Gráfico a la izquierda y su lectura a la derecha."""
    grafico = dbc.Card(dbc.CardBody([
        html.H4(titulo, className="tarjeta-titulo"),
        *(controles or []),
        dcc.Loading(dcc.Graph(id=id_grafico, config={"displaylogo": False, "responsive": True}),
                    type="dot", delay_show=400),
        html.Div(nota, className="tarjeta-nota"),
    ]), className="tarjeta")
    return dbc.Row([dbc.Col(grafico, lg=8), dbc.Col(html.Div(id=id_lectura), lg=4)], className="g-4")


def _opciones(variables):
    return [{"label": C.nombre(v), "value": v} for v in variables]


def _calidad():
    tabla = D.faltantes()
    faltan = (tarjeta_grafico(G.fig_faltantes(tabla), "Faltantes en los datos crudos",
                              "Datos crudos, antes del ETL: los filtros no aplican aquí porque las "
                              "variables filtradas se recodificaron en el ETL. Pasa el ratón para ver "
                              "qué se hizo con cada variable.")
              if tabla is not None else falta_artefacto("la tabla de faltantes (artifacts/eda_faltantes.csv)"))
    return html.Div([
        dbc.Row([
            dbc.Col(faltan, lg=8),
            dbc.Col(lectura([
                "El peso falta en casi todos los registros y el código de asegurador en cuatro de cada "
                "diez: se eliminaron.",
                "La especialidad y la raza se conservan con una categoría explícita para el faltante: "
                "la ausencia se asocia al desenlace, y la raza es un atributo protegido que no se imputa.",
                "«None» en HbA1c y glucosa no es un faltante: significa que la prueba no se hizo, y es "
                "la categoría más informativa de ese bloque.",
            ]), lg=4),
        ], className="g-4"),
        html.Div(className="mt-4"),
        _bloque("eda-atipicos", "eda-atipicos-lectura", "Valores atípicos de las variables numéricas",
                "Cajas de mediana y cuartiles; los bigotes llegan a 1,5 veces el rango intercuartílico."),
    ])


def layout():
    try:
        D.entrenamiento()
    except FileNotFoundError:
        return html.Div([encabezado("02 · EDA", "Análisis exploratorio", ""),
                         aviso("No se encontró data/processed/diabetes_train.parquet.", tipo="pendiente")])
    r = D.roles()
    todas = D.predictores()
    numericas = r["numericas"] + r["ordinales"]
    colores = ([{"label": "Sin color", "value": G.NINGUNO},
                {"label": "Readmisión", "value": G.Y}]
               + _opciones(r["binarias"] + r["ordinales"] + r["categoricas"]))
    return html.Div([
        encabezado("02 · EDA", "Análisis exploratorio: qué distingue a quien reingresa",
                   "Sobre el conjunto de entrenamiento; la prueba queda reservada para evaluar los modelos."),
        _filtros(),
        subpestanas([
            ("eda-objetivo", "Objetivo", _bloque(
                "eda-g-objetivo", "eda-l-objetivo", "Reparto del desenlace",
                "Encuentros por clase con los filtros aplicados.")),
            ("eda-univariado", "Univariado", _bloque(
                "eda-g-univariado", "eda-l-univariado", "Una variable y la readmisión",
                "Arriba, encuentros por valor y desenlace; abajo, la tasa de readmisión de cada valor. "
                "Las variables con muchos valores se agrupan en deciles.",
                [dcc.Dropdown(id="eda-var", options=_opciones(todas), value="number_inpatient",
                              clearable=False, className="mb-2")])),
            ("eda-bivariado", "Bivariado", _bloque(
                "eda-g-bivariado", "eda-l-bivariado", "Explorador bivariado",
                "Con dos numéricas, muestra de 4.000 encuentros con un leve temblor para separar los "
                "puntos; con la tasa en el eje Y, todos los encuentros.",
                [dbc.Row([
                    dbc.Col([html.Label("Eje X"), dcc.Dropdown(id="eda-x", options=_opciones(todas),
                                                              value="number_inpatient", clearable=False)], md=4),
                    dbc.Col([html.Label("Eje Y"), dcc.Dropdown(
                        id="eda-y", options=[{"label": "Tasa de readmisión (%)", "value": G.TASA}]
                        + _opciones(numericas), value=G.TASA, clearable=False)], md=4),
                    dbc.Col([html.Label("Color"), dcc.Dropdown(id="eda-color", options=colores,
                                                              value="age_ord", clearable=False)], md=4),
                ], className="g-2 mb-2")])),
            ("eda-correlacion", "Correlación", _bloque(
                "eda-g-correlacion", "eda-l-correlacion", "Matriz de correlación de Spearman",
                "Variables numéricas, ordinales y binarias, más el desenlace. Azul: relación inversa; "
                "rojo: directa; gris: ninguna.")),
            ("eda-calidad", "Calidad de los datos", _calidad()),
        ], SUBPESTANAS),
    ], className="pagina")


@callback(Output("eda-n", "children"), *FILTROS)
def contar(*filtros):
    df = D.filtrar(*filtros)
    total = len(D.entrenamiento())
    return (f"{miles(len(df))} encuentros seleccionados ({pct(100 * len(df) / total)} del "
            f"entrenamiento) · readmisión {pct(100 * df[G.Y].mean()) if len(df) else '–'}")


def _salida(resultado):
    figura, lineas = resultado[0], resultado[1]
    return figura, lectura(lineas) if lineas else None


@callback(Output("eda-g-objetivo", "figure"), Output("eda-l-objetivo", "children"),
          Input(SUBPESTANAS, "active_tab"), *FILTROS)
def objetivo(pestana, *filtros):
    if pestana != "eda-objetivo":
        return no_update, no_update
    return _salida(G.fig_objetivo(D.filtrar(*filtros)))


@callback(Output("eda-g-univariado", "figure"), Output("eda-l-univariado", "children"),
          Input(SUBPESTANAS, "active_tab"), Input("eda-var", "value"), *FILTROS)
def univariado(pestana, variable, *filtros):
    if pestana != "eda-univariado":
        return no_update, no_update
    return _salida(G.fig_univariado(D.filtrar(*filtros), variable))


@callback(Output("eda-g-bivariado", "figure"), Output("eda-l-bivariado", "children"),
          Input(SUBPESTANAS, "active_tab"), Input("eda-x", "value"), Input("eda-y", "value"),
          Input("eda-color", "value"), *FILTROS)
def bivariado(pestana, x, y, color, *filtros):
    if pestana != "eda-bivariado":
        return no_update, no_update
    if x == y:
        return G._vacio("Elige variables distintas para los dos ejes."), None
    return _salida(G.fig_bivariado(D.filtrar(*filtros), x, y, color))


@callback(Output("eda-g-correlacion", "figure"), Output("eda-l-correlacion", "children"),
          Input(SUBPESTANAS, "active_tab"), *FILTROS)
def correlacion(pestana, *filtros):
    if pestana != "eda-correlacion":
        return no_update, no_update
    return _salida(G.fig_correlacion(D.filtrar(*filtros)))


@callback(Output("eda-atipicos", "figure"), Output("eda-atipicos-lectura", "children"),
          Input(SUBPESTANAS, "active_tab"), *FILTROS)
def atipicos(pestana, *filtros):
    if pestana != "eda-calidad":
        return no_update, no_update
    return _salida(G.fig_atipicos(D.filtrar(*filtros)))
