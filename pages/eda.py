"""
Página 2 · EDA: el análisis exploratorio sobre el conjunto de entrenamiento.

Contenido mínimo de la sección 7.3.2 de la guía: filtros globales que
actualizan todos los gráficos, distribución del objetivo, univariado y un
explorador bivariado con ejes y color elegibles, correlación, faltantes y
atípicos, y una lectura junto a cada gráfico.

Además de ese mínimo, la página tiene vistas pensadas para un lector
clínico: la matriz de 100 pacientes (frecuencias naturales), el recorrido
del paciente (se eligen sus pasos con un clic y se lee su tasa), el
ranking de diagnósticos con su detalle, la pirámide de edad y sexo, la
curva de dosis-respuesta de la historia previa, el tratamiento (fármacos y
HbA1c), un comparador de dos perfiles y una matriz de riesgo de dos
variables. Todas responden a los filtros globales.

Cada gráfico se calcula solo cuando su subpestaña está activa: así no se
dibuja en un panel oculto (Plotly lo dejaría sin ancho) y un cambio de
filtro solo recalcula lo que se ve.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from src.tablero import analitica as A
from src.tablero import config as C
from src.tablero import datos as D
from src.tablero import graficos as G
from src.tablero import graficos_clinicos as GC
from src.tablero.componentes import (aviso, encabezado, falta_artefacto, icono, lectura,
                                     matriz_personas, subpestanas, tarjeta, tarjeta_grafico)
from src.tablero.formato import miles, num, pct

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
                     desplegable("f-sexo", _legibles("gender", o["sexo"]), "Todos")], md=2),
            dbc.Col([html.Label("Grupo racial", htmlFor="f-raza"),
                     desplegable("f-raza", _legibles("race", o["raza"]), "Todos")], md=2),
            dbc.Col([html.Label("Tipo de admisión", htmlFor="f-admision"),
                     desplegable("f-admision", _legibles("admission_type_id", o["admision"]), "Todos")], md=2),
            dbc.Col([html.Label("Días de hospitalización", htmlFor="f-dias"),
                     dcc.RangeSlider(id="f-dias", min=o["dias"][0], max=o["dias"][1], step=1,
                                     value=list(o["dias"]), allowCross=False,
                                     marks={d: str(d) for d in range(o["dias"][0], o["dias"][1] + 1, 3)},
                                     tooltip={"placement": "bottom"})], md=3),
        ], className="g-3"),
        html.Div(id="eda-n", className="filtros-n", **{"aria-live": "polite"}),
    ], titulo="Filtros",
        nota="El conjunto no tiene fechas, así que no hay filtro de periodo. Los filtros se "
             "aplican al conjunto de entrenamiento.", clase="tarjeta-filtros")


def _legibles(variable, valores):
    """Opciones de un desplegable: valor original, etiqueta en español."""
    return [{"label": C.nivel(variable, v), "value": v} for v in valores]


def _pulso():
    """Los filtros, contados en personas: «de cada 100 pacientes, 11 vuelven»."""
    return dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col(html.Div(id="eda-pulso-matriz"), md=5, lg=4),
        dbc.Col(html.Div(id="eda-pulso-texto"), md=7, lg=8, className="d-flex align-items-center"),
    ], className="g-3 align-items-center")), className="tarjeta tarjeta-personas mb-4")


def _bloque(id_grafico, id_lectura, titulo, nota, controles=None):
    """Gráfico a la izquierda y su lectura a la derecha."""
    grafico = dbc.Card(dbc.CardBody([
        html.H4(titulo, className="tarjeta-titulo"),
        *(controles or []),
        dcc.Loading(dcc.Graph(id=id_grafico, figure=G.figura_inicial(), config={"displayModeBar": False, "responsive": True}),
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


# --------------------------------------------------------------------------- #
# Vistas clínicas
# --------------------------------------------------------------------------- #
def _dosis():
    opciones = [{"label": C.nombre(v), "value": v} for v in A.DOSIS]
    return _bloque("eda-g-dosis", "eda-l-dosis", "Historia previa: más visitas, más reingresos",
                   "Tasa de readmisión según el valor de la variable, con su intervalo de confianza al 95 %. "
                   "La última categoría agrupa la cola (≥). Mueve el control para leer un valor concreto.",
                   [dbc.Row([
                       dbc.Col([html.Label("Variable", htmlFor="eda-dosis-var"),
                                dcc.Dropdown(id="eda-dosis-var", options=opciones, value="number_inpatient",
                                             clearable=False)], md=5),
                       dbc.Col([html.Label("Leer el valor", htmlFor="eda-dosis-valor"),
                                dcc.Slider(id="eda-dosis-valor", min=0, max=6, step=1, value=2,
                                           marks=None, tooltip=None)], md=7, className="control-simulador"),
                   ], className="g-2 mb-2")])


def _tratamiento():
    return html.Div([
        _bloque("eda-g-farmacos", "eda-l-farmacos", "Antidiabéticos: estado de la prescripción y readmisión",
                "Cada celda es la tasa de readmisión de los encuentros con ese fármaco en ese estado. "
                "Ámbar: por encima de la media; azul: por debajo. Las celdas con menos de 30 encuentros "
                "se muestran en gris con su conteo. Fármacos ordenados de más a menos prescritos; en la "
                "insulina, «dosis cambia» reúne las subidas y las bajadas."),
        html.Div(className="mt-4"),
        _bloque("eda-g-hba1c", "eda-l-hba1c", "HbA1c: medirla o no, y qué se hizo con el resultado",
                "Tasa de readmisión por resultado de la hemoglobina glicosilada, según si hubo cambio de la "
                "medicación antidiabética. Cada punto es la tasa y su segmento, el IC de Wilson al 95 %."),
    ])


def _comparador():
    o = D.opciones_filtro()
    diagnosticos = sorted(D.entrenamiento()["diag_1_grupo"].astype(str).unique())

    def grupo(sufijo, titulo, edades, clase):
        return dbc.Card(dbc.CardBody([
            html.H4(titulo, className="tarjeta-titulo"),
            html.Label("Edad", htmlFor=f"cmp-edad-{sufijo}"),
            dcc.Dropdown(id=f"cmp-edad-{sufijo}", options=o["edad"], value=edades, multi=True,
                         placeholder="Todas", className="mb-2"),
            html.Label("Sexo", htmlFor=f"cmp-sexo-{sufijo}"),
            dcc.Dropdown(id=f"cmp-sexo-{sufijo}", options=_legibles("gender", o["sexo"]), multi=True,
                         placeholder="Todos", className="mb-2"),
            html.Label("Hospitalizaciones el año previo", htmlFor=f"cmp-previas-{sufijo}"),
            dcc.RangeSlider(id=f"cmp-previas-{sufijo}", min=0, max=5, step=1, value=[0, 5],
                            marks={i: (str(i) if i < 5 else "5+") for i in range(6)}, allowCross=False),
            html.Label("Diagnóstico principal", htmlFor=f"cmp-diag-{sufijo}", className="mt-2"),
            dcc.Dropdown(id=f"cmp-diag-{sufijo}", options=diagnosticos, multi=True, placeholder="Todos"),
            html.Div(id=f"cmp-n-{sufijo}", className="filtros-n"),
        ]), className=f"tarjeta {clase}")

    return html.Div([
        dbc.Row([
            dbc.Col(grupo("a", "Grupo A", [0, 1, 2, 3, 4], "grupo-perfil"), md=6, lg=3),
            dbc.Col(grupo("b", "Grupo B", [7, 8, 9], "grupo-perfil-b"), md=6, lg=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4("Perfil de los dos grupos", className="tarjeta-titulo"),
                dcc.Loading(dcc.Graph(id="eda-g-comparador", figure=G.figura_inicial(), config={"displayModeBar": False, "responsive": True}),
                            type="dot", delay_show=400),
                html.Div("Porcentaje de encuentros de cada grupo que cumple cada condición. Los dos grupos "
                         "se toman dentro de los filtros generales de la página.", className="tarjeta-nota"),
            ]), className="tarjeta"), lg=6),
        ], className="g-4"),
        html.Div(id="eda-l-comparador", className="mt-4"),
    ])


def _grafico(id_, alto=None):
    """Gráfico de la página: arranca en blanco y sin barra de herramientas."""
    # style solo si hay altura: dcc.Graph falla con style=None (lee style.height).
    extra = {"style": {"height": f"{alto}px"}} if alto else {}
    return dcc.Graph(id=id_, figure=G.figura_inicial(), config={"displayModeBar": False, "responsive": True},
                     **extra)


def _recorrido():
    """El recorrido del paciente como tres etapas en las que se elige con un clic.

    Sustituye al diagrama de Sankey: en lugar de seguir cintas cruzadas, se
    eligen los pasos de un paciente y se lee directamente su tasa.
    """
    etapas = []
    for i, (etapa, (_, titulo, icono_e)) in enumerate(A.ETAPAS.items(), start=1):
        etapas.append(dbc.Col(html.Div([
            html.Div([
                html.Span(icono_e, className="material-symbols-rounded etapa-icono", **{"aria-hidden": "true"}),
                html.Div([html.Div(f"Paso {i}", className="etapa-paso"), html.Div(titulo, className="etapa-titulo")]),
                html.Div(id=f"rec-chip-{etapa}", className="etapa-chip"),
            ], className="etapa-cabeza"),
            dcc.Loading(_grafico(f"rec-g-{etapa}"), type="dot", delay_show=400),
        ], className="etapa tarjeta"), lg=4, className="etapa-col"))
    return html.Div([
        dcc.Store(id="rec-eleccion", data={e: A.CUALQUIERA for e in A.ETAPAS}),
        html.Div(id="rec-resultado"),
        html.Div([
            html.P([icono("touch_app", "me-1"), "Haz clic en una barra para elegir ese paso del recorrido; otro clic "
                    "sobre la misma lo quita. Ámbar: más readmisión que la media; azul: menos."],
                   className="nota-fuente m-0"),
            html.Button([icono("restart_alt", "me-1"), "Reiniciar recorrido"], id="rec-reiniciar",
                        className="boton-terciario", n_clicks=0),
        ], className="d-flex flex-wrap justify-content-between align-items-center gap-2 mt-3 mb-2"),
        dbc.Row(etapas, className="g-3 recorrido"),
    ])


def _diagnosticos():
    """Ranking de diagnósticos principales; al hacer clic, sus secundarios."""
    return html.Div([
        dcc.Store(id="diag-principal"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4("Diagnóstico principal", className="tarjeta-titulo"),
                dcc.Loading(_grafico("diag-g-principal"), type="dot", delay_show=400),
                html.Div([icono("touch_app", "me-1"), "Haz clic en un diagnóstico para ver, a la derecha, cómo cambia "
                          "el riesgo según el diagnóstico secundario."], className="tarjeta-nota"),
            ]), className="tarjeta"), lg=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H4(id="diag-titulo-secundario", className="tarjeta-titulo"),
                dcc.Loading(_grafico("diag-g-secundario"), type="dot", delay_show=400),
                html.Div("Capítulos ICD-9 con al menos 50 encuentros dentro del diagnóstico principal elegido. "
                         "Las barras finas son el IC de Wilson al 95 %.", className="tarjeta-nota"),
            ]), className="tarjeta"), lg=6),
        ], className="g-4 diag-clic"),
        html.Div(id="diag-lectura", className="mt-4"),
    ])


def _matriz():
    opciones = [{"label": C.nombre(v), "value": v} for v in A.MATRIZ]
    return html.Div([
        dbc.Card(dbc.CardBody([
            html.H4("Matriz de riesgo: dos características a la vez", className="tarjeta-titulo"),
            dbc.Row([
                dbc.Col([html.Label("Filas", htmlFor="mat-fila"),
                         dcc.Dropdown(id="mat-fila", options=opciones, value="number_inpatient", clearable=False)], md=5),
                dbc.Col(html.Button([icono("swap_horiz")], id="mat-intercambiar", className="boton-icono",
                                    title="Intercambiar filas y columnas", n_clicks=0,
                                    **{"aria-label": "Intercambiar filas y columnas"}),
                        md=2, className="d-flex align-items-end justify-content-center"),
                dbc.Col([html.Label("Columnas", htmlFor="mat-columna"),
                         dcc.Dropdown(id="mat-columna", options=opciones, value="age_ord", clearable=False)], md=5),
            ], className="g-2 mb-3"),
            dcc.Loading(_grafico("eda-g-matriz"), type="dot", delay_show=400),
            html.Div("Cada celda es la tasa de readmisión de los encuentros que tienen a la vez esas dos "
                     "características. Ámbar: por encima de la media; azul: por debajo.", className="tarjeta-nota"),
        ]), className="tarjeta"),
        html.Div(id="eda-l-matriz", className="mt-4"),
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
        _pulso(),
        subpestanas([
            ("eda-objetivo", "Objetivo", _bloque(
                "eda-g-objetivo", "eda-l-objetivo", "Reparto del desenlace",
                "Encuentros por clase con los filtros aplicados.")),
            ("eda-recorrido", "Recorrido del paciente", _recorrido()),
            ("eda-diagnosticos", "Diagnósticos", _diagnosticos()),
            ("eda-demografia", "Edad y sexo", _bloque(
                "eda-g-piramide", "eda-l-piramide", "Pirámide de edad y sexo",
                "Encuentros por decenio de edad; el color y la etiqueta de cada barra son su tasa de "
                "readmisión.")),
            ("eda-dosis", "Historia previa", _dosis()),
            ("eda-tratamiento", "Tratamiento", _tratamiento()),
            ("eda-comparador", "Comparar perfiles", _comparador()),
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
            ("eda-matriz", "Matriz de riesgo", _matriz()),
            ("eda-correlacion", "Correlación", _bloque(
                "eda-g-correlacion", "eda-l-correlacion", "Matriz de correlación de Spearman",
                "Variables numéricas, ordinales y binarias, más el desenlace. Azul: relación inversa; "
                "ámbar: directa; gris: ninguna.")),
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


# --------------------------------------------------------------------------- #
# Callbacks de las vistas clínicas
# --------------------------------------------------------------------------- #
@callback(Output("eda-pulso-matriz", "children"), Output("eda-pulso-texto", "children"), *FILTROS)
def pulso(*filtros):
    df = D.filtrar(*filtros)
    if df.empty:
        return None, html.P("No hay encuentros con estos filtros.", className="cuerpo")
    tasa = 100 * df[G.Y].mean()
    vuelven = A.personas_de_100(tasa)
    general = 100 * D.entrenamiento()[G.Y].mean()
    matriz = matriz_personas(
        [{"n": vuelven, "clase": "p-readmitido", "etiqueta": "vuelven antes de 30 días"},
         {"n": 100 - vuelven, "clase": "p-neutro", "etiqueta": "no vuelven", "relleno": False}],
        f"De cada 100 pacientes seleccionados, {vuelven} son readmitidos antes de 30 días.")
    comparacion = ("igual que" if abs(tasa - general) < 0.25 else
                   ("más que" if tasa > general else "menos que"))
    texto = html.Div([
        html.Div([html.Span(f"{vuelven}", className="cifra-grande"),
                  html.Span(" de cada 100", className="cifra-grande", style={"fontWeight": 400})]),
        html.P(f"pacientes con este perfil vuelven al hospital antes de 30 días ({pct(tasa)}, "
               f"{comparacion} el {pct(general)} de todo el entrenamiento).", className="cuerpo mb-1"),
        html.P(f"Basado en {miles(len(df))} encuentros. Cambia los filtros de arriba y la cifra se actualiza.",
               className="tarjeta-nota mb-0"),
    ])
    return matriz, texto


def _prevalencia(df):
    return 100 * df[G.Y].mean() if len(df) else 0.0


@callback(Output("rec-eleccion", "data"),
          Input("rec-g-admision", "clickData"), Input("rec-g-estancia", "clickData"),
          Input("rec-g-alta", "clickData"), Input("rec-reiniciar", "n_clicks"),
          State("rec-eleccion", "data"), prevent_initial_call=True)
def elegir_paso(c_adm, c_est, c_alta, _reiniciar, eleccion):
    """Un clic en una barra elige ese paso; otro clic en la misma lo quita."""
    disparador = ctx.triggered_id
    eleccion = dict(eleccion or {e: A.CUALQUIERA for e in A.ETAPAS})
    if disparador == "rec-reiniciar":
        return {e: A.CUALQUIERA for e in A.ETAPAS}
    clic = {"rec-g-admision": ("admision", c_adm), "rec-g-estancia": ("estancia", c_est),
            "rec-g-alta": ("alta", c_alta)}.get(disparador)
    if not clic or not clic[1]:
        return no_update
    etapa, datos = clic
    nivel = datos["points"][0]["y"]
    eleccion[etapa] = A.CUALQUIERA if eleccion.get(etapa) == nivel else nivel
    return eleccion


def _chip(etapa, nivel):
    if nivel in (None, A.CUALQUIERA):
        return html.Span("Cualquiera", className="chip chip-vacio")
    return html.Span([icono("check", "me-1"), nivel], className="chip chip-activo")


@callback(Output("rec-g-admision", "figure"), Output("rec-g-estancia", "figure"), Output("rec-g-alta", "figure"),
          Output("rec-chip-admision", "children"), Output("rec-chip-estancia", "children"),
          Output("rec-chip-alta", "children"), Output("rec-resultado", "children"),
          Input(SUBPESTANAS, "active_tab"), Input("rec-eleccion", "data"), *FILTROS)
def recorrido(pestana, eleccion, *filtros):
    if pestana != "eda-recorrido":
        return (no_update,) * 7
    eleccion = eleccion or {e: A.CUALQUIERA for e in A.ETAPAS}
    df = D.filtrar(*filtros)
    if df.empty:
        return G._vacio(), G._vacio(), G._vacio(), None, None, None, None
    prevalencia = _prevalencia(df)
    resumen = A.resumen_etapas(df)
    figuras = [GC.fig_etapa(resumen[e], prevalencia, eleccion.get(e)) for e in A.ETAPAS]
    chips = [_chip(e, eleccion.get(e)) for e in A.ETAPAS]
    r = A.tasa_recorrido(df, eleccion)
    pasos = [eleccion.get(e) for e in A.ETAPAS]
    camino = [html.Span(p if p != A.CUALQUIERA else "cualquiera", className="camino-paso" +
                        ("" if p != A.CUALQUIERA else " camino-libre")) for p in pasos]
    for k in (2, 1):
        camino.insert(k, html.Span(icono("arrow_forward"), className="camino-flecha"))
    if r["n"] < 30:
        cuerpo = [html.Div(camino, className="camino"),
                  aviso(f"Solo {miles(r['n'])} encuentros siguieron este recorrido con los filtros actuales: "
                        "demasiado pocos para dar una tasa fiable. Quita algún paso.", tipo="pendiente")]
        matriz = None
    else:
        vuelven = A.personas_de_100(r["tasa"])
        relacion = r["tasa"] / prevalencia if prevalencia else np.nan
        comparacion = ("como la media" if abs(relacion - 1) < 0.05 else
                       f"{num(relacion, 1)} veces la media" if relacion > 1 else f"{pct(100 * (1 - relacion), 0)} menos que la media")
        matriz = matriz_personas(
            [{"n": vuelven, "clase": "p-readmitido", "etiqueta": "vuelven antes de 30 días"},
             {"n": 100 - vuelven, "clase": "p-neutro", "etiqueta": "no vuelven", "relleno": False}],
            f"De cada 100 pacientes con este recorrido, {vuelven} son readmitidos antes de 30 días.")
        cuerpo = [html.Div(camino, className="camino"),
                  html.Div([html.Span(f"{vuelven}", className="cifra-grande"),
                            html.Span(" de cada 100", className="cifra-grande", style={"fontWeight": 400})]),
                  html.P(f"pacientes con este recorrido vuelven antes de 30 días: {pct(r['tasa'])} "
                         f"(IC 95 %: {num(r['ic_bajo'], 1)}–{num(r['ic_alto'], 1)} %), {comparacion} "
                         f"({pct(prevalencia)}).", className="cuerpo mb-1"),
                  html.P(f"Basado en {miles(r['n'])} encuentros del entrenamiento.", className="tarjeta-nota mb-2")]
    resultado = dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col(matriz, md=5, lg=4) if matriz is not None else None,
        dbc.Col(cuerpo, md=7 if matriz is not None else 12, lg=8 if matriz is not None else 12),
    ], className="g-3 align-items-center")), className="tarjeta tarjeta-personas")
    return (*figuras, *chips, resultado)


@callback(Output("diag-principal", "data"), Input("diag-g-principal", "clickData"), prevent_initial_call=True)
def elegir_diagnostico(clic):
    return clic["points"][0]["y"] if clic else no_update


@callback(Output("diag-g-principal", "figure"), Output("diag-g-secundario", "figure"),
          Output("diag-titulo-secundario", "children"), Output("diag-lectura", "children"),
          Input(SUBPESTANAS, "active_tab"), Input("diag-principal", "data"), *FILTROS)
def diagnosticos(pestana, principal, *filtros):
    if pestana != "eda-diagnosticos":
        return (no_update,) * 4
    df = D.filtrar(*filtros)
    if df.empty:
        return G._vacio(), G._vacio(), "Diagnóstico secundario", None
    prevalencia = _prevalencia(df)
    ranking = A.ranking_diagnosticos(df)
    if ranking.empty:
        return G._vacio(), G._vacio(), "Diagnóstico secundario", None
    frecuente = ranking.sort_values("n", ascending=False)["nivel"].iloc[0]
    principal = principal if principal in set(ranking["nivel"]) else frecuente
    fila = ranking.set_index("nivel").loc[principal]
    sec = A.secundarios(df, principal)
    prev_principal = float(fila["tasa"])
    figura_sec = GC.fig_ranking_diagnosticos(sec, prev_principal,
                                             titulo_x=f"Readmisión (%) · la línea es la tasa de «{principal}»")
    alto, bajo = ranking.iloc[0], ranking.iloc[-1]
    lineas = [f"Por diagnóstico principal, la readmisión va del {pct(bajo['tasa'])} («{bajo['nivel']}») al "
              f"{pct(alto['tasa'])} («{alto['nivel']}»), con una media de {pct(prevalencia)}.",
              f"«{principal}» ({miles(fila['n'])} encuentros) reingresa en el {pct(prev_principal)} de los casos."]
    if len(sec) >= 2:
        lineas.append(f"Dentro de «{principal}», el secundario de más riesgo es «{sec.iloc[0]['nivel']}» "
                      f"({pct(sec.iloc[0]['tasa'])}) y el de menos, «{sec.iloc[-1]['nivel']}» "
                      f"({pct(sec.iloc[-1]['tasa'])}).")
    return (GC.fig_ranking_diagnosticos(ranking, prevalencia, principal), figura_sec,
            [html.Span("Secundarios dentro de ", className="titulo-ligero"), f"«{principal}»"], lectura(lineas))


@callback(Output("eda-g-piramide", "figure"), Output("eda-l-piramide", "children"),
          Input(SUBPESTANAS, "active_tab"), *FILTROS)
def piramide(pestana, *filtros):
    if pestana != "eda-demografia":
        return no_update, no_update
    return _salida(GC.fig_piramide(A.piramide(D.filtrar(*filtros))))


@callback(Output("eda-dosis-valor", "max"), Output("eda-dosis-valor", "marks"),
          Output("eda-dosis-valor", "value"), Input("eda-dosis-var", "value"))
def dosis_control(variable):
    tabla = A.dosis_respuesta(D.entrenamiento(), variable)
    etiquetas = tabla["etiqueta"].tolist()
    salto = max(1, len(etiquetas) // 8)
    marcas = {i: e for i, e in enumerate(etiquetas) if i % salto == 0 or i == len(etiquetas) - 1}
    return len(etiquetas) - 1, marcas, min(2, len(etiquetas) - 1)


@callback(Output("eda-g-dosis", "figure"), Output("eda-l-dosis", "children"),
          Input(SUBPESTANAS, "active_tab"), Input("eda-dosis-var", "value"),
          Input("eda-dosis-valor", "value"), *FILTROS)
def dosis(pestana, variable, valor, *filtros):
    if pestana != "eda-dosis":
        return no_update, no_update
    df = D.filtrar(*filtros)
    if df.empty:
        return G._vacio(), None
    return _salida(GC.fig_dosis(A.dosis_respuesta(df, variable), variable, valor, _prevalencia(df)))


@callback(Output("eda-g-farmacos", "figure"), Output("eda-l-farmacos", "children"),
          Output("eda-g-hba1c", "figure"), Output("eda-l-hba1c", "children"),
          Input(SUBPESTANAS, "active_tab"), *FILTROS)
def tratamiento(pestana, *filtros):
    if pestana != "eda-tratamiento":
        return no_update, no_update, no_update, no_update
    df = D.filtrar(*filtros)
    if df.empty:
        return G._vacio(), None, G._vacio(), None
    prevalencia = _prevalencia(df)
    farmacos = _salida(GC.fig_farmacos(A.farmacos(df), prevalencia))
    hba1c = _salida(GC.fig_hba1c(A.hba1c(df), prevalencia))
    return (*farmacos, *hba1c)


def _nombre_grupo(edades, sexos, previas, diag, letra):
    """Nombre corto del grupo para la leyenda («A: 70+ años, hombres»)."""
    partes = []
    if edades:
        e = sorted(int(x) for x in edades)
        desde, hasta = A.edad_legible(e[0]).split("–")[0], A.edad_legible(e[-1]).split("–")[-1]
        partes.append(f"{desde}–{hasta} años")
    if sexos and len(sexos) == 1:
        partes.append("mujeres" if sexos[0] == "Female" else "hombres")
    if previas and list(previas) != [0, 5]:
        partes.append(f"{previas[0]}–{'5+' if previas[1] >= 5 else previas[1]} hosp. previas")
    if diag:
        partes.append(diag[0].lower() + (" y otros" if len(diag) > 1 else ""))
    return f"{letra}: " + (", ".join(partes) if partes else "todos")


@callback(Output("eda-g-comparador", "figure"), Output("eda-l-comparador", "children"),
          Output("cmp-n-a", "children"), Output("cmp-n-b", "children"),
          Input(SUBPESTANAS, "active_tab"),
          Input("cmp-edad-a", "value"), Input("cmp-sexo-a", "value"), Input("cmp-previas-a", "value"),
          Input("cmp-diag-a", "value"), Input("cmp-edad-b", "value"), Input("cmp-sexo-b", "value"),
          Input("cmp-previas-b", "value"), Input("cmp-diag-b", "value"), *FILTROS)
def comparar(pestana, ea, sa, pa, da, eb, sb, pb, db, *filtros):
    if pestana != "eda-comparador":
        return no_update, no_update, no_update, no_update
    base = D.filtrar(*filtros)
    grupo_a = A.filtrar_perfil(base, ea, sa, pa, da)
    grupo_b = A.filtrar_perfil(base, eb, sb, pb, db)
    nombre_a, nombre_b = _nombre_grupo(ea, sa, pa, da, "A"), _nombre_grupo(eb, sb, pb, db, "B")
    figura, lineas = GC.fig_comparador(A.perfil(grupo_a), A.perfil(grupo_b), nombre_a, nombre_b)
    conteo = lambda g: f"{miles(len(g))} encuentros" + ("" if len(g) >= 100 else " · pocos: cifras inestables")
    return figura, (lectura(lineas) if lineas else None), conteo(grupo_a), conteo(grupo_b)


@callback(Output("mat-fila", "value"), Output("mat-columna", "value"),
          Input("mat-intercambiar", "n_clicks"), State("mat-fila", "value"), State("mat-columna", "value"),
          prevent_initial_call=True)
def intercambiar(_, fila, columna):
    return columna, fila


@callback(Output("eda-g-matriz", "figure"), Output("eda-l-matriz", "children"), Input(SUBPESTANAS, "active_tab"),
          Input("mat-fila", "value"), Input("mat-columna", "value"), *FILTROS)
def matriz(pestana, fila, columna, *filtros):
    if pestana != "eda-matriz":
        return no_update, no_update
    if fila == columna:
        return G._vacio("Elige dos variables distintas."), None
    df = D.filtrar(*filtros)
    if df.empty:
        return G._vacio(), None
    tasa, n = A.matriz_riesgo(df, fila, columna)
    return _salida(GC.fig_matriz_riesgo(tasa, n, fila, columna, _prevalencia(df)))
