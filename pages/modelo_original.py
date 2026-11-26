"""
Página «Modelo original» (/modelo-original): DES-CPL contado como un relato.

Siete pasos, cada uno con un título que da la conclusión, una visual y su
lectura calculada con los datos:

1. La idea: un comité de modelos y un paciente que llega.
2. Cómo decide, con un paciente real (la pieza central): probabilidades de los
   4 modelos, región de competencia, competencia local y global, y control de
   λ y τ que recalcula los pesos en vivo.
3. El resultado: AUC-PR por fold y diferencias con intervalo, las tres pruebas
   del protocolo y la regla de superioridad.
4. Por qué empata: λ por fold, ablación y diagnóstico.
5. Generalización y estabilidad: rango medio, robustez y semillas.
6. Las 13 comprobaciones posteriores al protocolo.
7. Veredicto.

Todo se calcula de forma perezosa: el contenido de un paso se construye solo
cuando ese paso está activo, y el explicador carga el artefacto al pedir el
primer paciente. Si falta un CSV, su sección muestra un aviso y el resto
funciona. La lógica está en ``src/tablero`` (``explicador``, ``evidencia``,
``graficos_relato``, ``ilustracion``); aquí solo se arma la interfaz.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, callback, clientside_callback, ctx, dcc, html, no_update

from src.tablero import analitica as A
from src.tablero import config as C
from src.tablero import datos as D
from src.tablero import evidencia as E
from src.tablero import explicador as X
from src.tablero import graficos as G
from src.tablero import graficos_relato as R
from src.tablero.componentes import (aviso, encabezado, fila_kpi, icono, kpi, lectura, tabla, tarjeta,
                                     tarjeta_grafico, tex)
from src.tablero.formato import decimales_es, miles, num, p_valor, pct
from src.tablero.ilustracion import svg_comite
from src.tablero.tema import color_modelo, papel_modelo

dash.register_page(__name__, path="/modelo-original", name="Modelo original", order=3,
                   title="Modelo original · Readmisión temprana")

PASOS = [("idea", "La idea"), ("decide", "Cómo decide"), ("resultado", "El resultado"),
         ("empata", "Por qué empata"), ("generaliza", "Generalización"),
         ("comprobaciones", "Comprobaciones"), ("veredicto", "Veredicto")]
IDS = [p[0] for p in PASOS]


# =========================================================================== #
# Piezas comunes
# =========================================================================== #
def _titulo(texto: str):
    """Título del paso: dice la conclusión, no el tema."""
    return html.H2(texto, className="paso-relato-titulo")


def _sin_datos(que: str, archivo: str):
    return aviso(f"Falta {archivo}: {que} no se puede mostrar. El resto de la página sigue funcionando.",
                 "Sección sin datos", "pendiente")


def _hiperparametros() -> dict:
    """k, τ, λ y poda del artefacto, leídos de su metadata (sin cargar el modelo)."""
    try:
        h = dict(D.metadata(C.MODELO_ORIGINAL)["hiperparametros"])
    except (FileNotFoundError, KeyError):
        h = {}
    return {"k": h.get("k", 20), "tau": h.get("tau", 0.0003), "lam": h.get("encogimiento", 0.9),
            "poda": h.get("poda", 4)}


def _barras(filas, destacar=None, marca=None, pie=None):
    """Barras horizontales HTML (nombre, pista, valor). El ancho se anima por CSS al cambiar.

    ``filas`` son tuplas (nombre, fracción 0-1, texto). Cada barra lleva el color del
    papel de su modelo (``tema.papel_modelo``), como en los gráficos de Plotly.
    ``marca`` es una posición 0-1 y su rótulo (una línea vertical, como el umbral).
    """
    cuerpo = []
    for i, (nombre, fraccion, texto) in enumerate(filas):
        relleno = html.Div(className="barra-relleno", style={"width": f"{100 * max(0.0, min(1.0, fraccion)):.2f}%"})
        pista = [relleno]
        if marca is not None:
            pista.append(html.Div(className="barra-umbral", style={"left": f"{100 * marca[0]:.2f}%"}, title=marca[1]))
        cuerpo.append(html.Div([html.Span(nombre, className="barra-nombre"),
                                html.Div(pista, className="barra-pista"),
                                html.Span(texto, className="barra-valor")],
                               className=f"barra-fila barra-{papel_modelo(nombre)}"
                                         + (" barra-destacada" if destacar == i else "")))
    if pie:
        cuerpo.append(html.Div(pie, className="tarjeta-nota"))
    return html.Div(cuerpo, className="barras")


def _tabla_corta(encabezados, filas):
    return html.Div(tabla(encabezados, filas), className="tabla-corta")


# =========================================================================== #
# 1 · La idea
# =========================================================================== #
def paso_idea():
    h = _hiperparametros()
    return html.Div([
        _titulo("DES-CPL deja que cada paciente decida en qué modelos confiar"),
        dbc.Row([
            dbc.Col(tarjeta(dcc.Markdown(svg_comite(), dangerously_allow_html=True, className="il-contenedor"),
                            titulo="Un comité de modelos y un paciente que llega",
                            nota="Ilustración con el lenguaje de la portada. Las barras son ilustrativas: "
                                 "no representan datos reales."), lg=7),
            dbc.Col([
                tarjeta([
                    html.P("Para cada paciente, mira cómo le fue a cada modelo con los pacientes más parecidos y "
                           "da más peso al que mejor estimó su riesgo.", className="frase-clave"),
                    html.Ol([
                        html.Li([html.Strong("Región. "), f"Los {h['k']} vecinos readmitidos y los {h['k']} no "
                                 "readmitidos más cercanos, en el espacio de las probabilidades de los modelos."]),
                        html.Li([html.Strong("Competencia. "), "El error de Brier de cada modelo en esa región."]),
                        html.Li([html.Strong("Mezcla. "), f"Pesos que bajan con el error, mezclados con la competencia "
                                 f"global según λ = {num(h['lam'], 1)}, porque la local es ruidosa."]),
                    ], className="pasos-lista"),
                ], titulo="En una frase"),
                lectura([
                    f"Combina {h['poda']} de los 7 modelos base (los de mayor AUC-PR fuera de pliegue).",
                    "Es original en readmisión de diabéticos: no se encontró selección dinámica de ensambles "
                    "aplicada a este problema (capítulo 14).",
                ], titulo="Qué lo distingue"),
            ], lg=5, className="d-grid gap-3 align-content-start"),
        ], className="g-4"),
    ])


# =========================================================================== #
# 2 · Cómo decide
# =========================================================================== #
def _marcas_tau():
    h = _hiperparametros()
    destacado = _indice_tau(h["tau"])
    marcas = {4: "0,001", 7: "0,01", 10: "1"}
    marcas[destacado] = f"{num(C.TAUS[destacado], 4)} · artefacto"
    return marcas


def _indice_tau(tau: float) -> int:
    return int(np.argmin([abs(np.log10(t) - np.log10(tau)) for t in C.TAUS]))


#: Las ecuaciones del método (capítulo 15), con la misma notación del libro.
#: Van en texto LaTeX para que MathJax las componga: la notación escrita a
#: mano (subíndices, Σ) se lee mal proyectada.
_FORMULAS_METODO = r"""
**1 · Prevalencia local suavizada** (2k vecinos sin estratificar, s = 10 pseudo-observaciones hacia la global $\pi$)

$$\hat q(x) = \frac{n^{+}(x) + s\,\pi}{2k + s}$$

**2 · Competencia local:** Brier de los $k$ vecinos readmitidos ($R^{+}$) y de los $k$ no readmitidos ($R^{-}$), ponderados por esa prevalencia

$$\hat C_m^{\,\mathrm{loc}}(x) = \hat q(x)\,\frac{1}{k}\sum_{i \in R^{+}} (1 - p_{m,i})^2 + \bigl(1 - \hat q(x)\bigr)\,\frac{1}{k}\sum_{i \in R^{-}} p_{m,i}^2$$

**3 · Encogimiento** hacia la competencia global del modelo

$$\hat C_m(x) = (1 - \lambda)\,\hat C_m^{\,\mathrm{loc}}(x) + \lambda\, C_m^{\,\mathrm{glob}}$$

**4 · Pesos y predicción**

$$w_m(x) = \frac{\exp\bigl(-\hat C_m(x)/\tau\bigr)}{\sum_j \exp\bigl(-\hat C_j(x)/\tau\bigr)}, \qquad \hat p(x) = \sum_m w_m(x)\, p_m(x)$$
"""

_FORMULAS_PROPIEDADES = r"""
**Regla propia (Proposición 2).** En una zona con riesgo verdadero $q$, un modelo que predice $p$ tiene Brier esperado

$$C_q(p) = q(1 - p)^2 + (1 - q)\,p^2 = q(1 - q) + (p - q)^2$$

Su derivada, $2(p - q)$, se anula en $p = q$ y la segunda derivada es $2 > 0$: gana el modelo calibrado en esa zona. Con $\tfrac12$ por clase el mínimo queda en $p = 0.5$ sea cual sea $q$, y se premia al modelo de probabilidades más altas.

**λ óptimo (Proposición 4).** Con varianza local $v$ y heterogeneidad real $b^2$:

$$\mathrm{ECM}(\lambda) = (1 - \lambda)^2\, v + \lambda^2\, b^2 \;\Rightarrow\; \lambda^{\ast} = \frac{v}{v + b^2}$$

**Casos límite.** $\lambda = 1$: pesos globales. $\tau \to \infty$: promedio simple. $\tau \to 0$: selección dura del mejor modelo local.
"""


def _formulas():
    """Las cuatro ecuaciones del método y las dos propiedades que lo justifican.

    Van junto al simulador para poder señalar cada término mientras se mueven
    λ y τ: el criterio de fundamentos teóricos se sustenta desde aquí.
    """
    return tarjeta(dbc.Row([
        dbc.Col(tex(_FORMULAS_METODO, "formulas"), lg=7),
        dbc.Col(tex(_FORMULAS_PROPIEDADES, "formulas"), lg=5),
    ], className="g-4"), titulo="Las fórmulas detrás de lo que acabas de ver",
        nota="Notación del capítulo 15 del libro; las demostraciones completas están ahí.")


def paso_decide():
    if not D.modelos_disponibles() or C.MODELO_ORIGINAL not in D.modelos_disponibles():
        return html.Div([_titulo("Cómo decide DES-CPL con un paciente real"),
                         _sin_datos("el explicador", "artifacts/des_cpl.joblib")])
    h = _hiperparametros()
    n = X.n_pacientes_prueba()
    inicial = X.indice_al_azar(2026)
    return html.Div([
        _titulo(f"Con λ = {num(h['lam'], 1)}, casi todo el peso viene de la competencia global; "
                "la región del paciente solo afina"),
        dcc.Store(id="mo-caso"),
        dbc.Row([
            dbc.Col(tarjeta([
                html.Label("Paciente de prueba (nº)", htmlFor="mo-indice"),
                dbc.Input(id="mo-indice", type="number", min=1, max=n, step=1, value=inicial + 1, debounce=True),
                html.Div([
                    dbc.Button([icono("shuffle", "me-2"), "Al azar"], id="mo-azar", className="boton-secundario",
                               n_clicks=0),
                    dbc.Button([icono("emergency", "me-2"), "Uno readmitido"], id="mo-azar-1",
                               className="boton-secundario", n_clicks=0),
                ], className="botones"),
                html.Div(id="mo-ficha", className="ficha-paciente", **{"aria-live": "polite"}),
            ], titulo="Elige un paciente", nota=f"Hay {miles(n)} pacientes en el conjunto de prueba, ninguno visto "
                                                "al entrenar."), lg=4),
            dbc.Col(dcc.Loading(tarjeta(html.Div(id="mo-probs"), titulo="Lo que dice cada modelo"),
                                type="dot", delay_show=300), lg=8),
        ], className="g-4"),
        dbc.Row([
            dbc.Col(dcc.Loading(tarjeta_grafico(G.figura_inicial(), "Su región de competencia", id_="mo-fig-proyeccion", alto=380,
                                                nota="Los dos primeros componentes de los logits de los modelos "
                                                     "(calculados una vez). Diamante: este paciente; círculos: "
                                                     "sus vecinos; gris: pacientes de selección."),
                                type="dot", delay_show=300), lg=6),
            dbc.Col(tarjeta_grafico(G.figura_inicial(), "Competencia local frente a global", id_="mo-fig-competencias", alto=300,
                                    nota="Brier de cada modelo medido sobre el mejor de la misma medida (menos es "
                                         "mejor). El rombo es la competencia que se usa con el λ elegido."), lg=6),
        ], className="g-4 mt-1"),
        _formulas(),
        tarjeta([
            dbc.Row([
                dbc.Col([
                    html.Label(["λ: peso de la competencia global ", html.Span(id="mo-lambda-valor", className="valor-control")],
                               htmlFor="mo-lambda"),
                    dcc.Slider(id="mo-lambda", min=0, max=1, step=0.05, value=h["lam"], updatemode="drag",
                               marks={0: "0", 0.5: "0,5", 1: "1", h["lam"]: f"{num(h['lam'], 1)} · artefacto"}),
                    html.Label(["τ: temperatura de los pesos ", html.Span(id="mo-tau-valor", className="valor-control")],
                               htmlFor="mo-tau", className="mt-4"),
                    dcc.Slider(id="mo-tau", min=0, max=len(C.TAUS) - 1, step=1, value=_indice_tau(h["tau"]),
                               updatemode="drag", marks=_marcas_tau(), allow_direct_input=False),
                    dbc.Button([icono("restart_alt", "me-2"), "Volver al artefacto"], id="mo-restablecer",
                               className="boton-secundario mt-4", n_clicks=0),
                ], lg=5, className="control-simulador"),
                dbc.Col([
                    html.Div("Peso que recibe cada modelo", className="etiqueta-mini"),
                    html.Div(id="mo-pesos", **{"aria-live": "polite"}),
                    html.Div(id="mo-final", className="resultado-final", **{"aria-live": "polite"}),
                ], lg=7),
            ], className="g-4"),
        ], titulo="Mueve λ y τ: los pesos y la probabilidad se recalculan al instante",
            nota="Solo se recombina la competencia ya calculada: no se vuelven a buscar vecinos."),
        html.Div(id="mo-lectura-decide", className="mt-3"),
    ])


@callback(Output("mo-indice", "value"),
          Input("mo-azar", "n_clicks"), Input("mo-azar-1", "n_clicks"), prevent_initial_call=True)
def elegir_paciente(n_azar, n_readmitido):
    """Un paciente al azar (o un readmitido al azar) con una semilla distinta en cada clic."""
    solo_readmitidos = ctx.triggered_id == "mo-azar-1"
    semilla = (n_azar or 0) * 7919 + (n_readmitido or 0) * 104729 + 1
    return X.indice_al_azar(semilla, solo_readmitidos) + 1


@callback(Output("mo-caso", "data"), Output("mo-ficha", "children"), Output("mo-probs", "children"),
          Output("mo-fig-proyeccion", "figure"), Input("mo-indice", "value"))
def calcular_paciente(numero):
    """Lo caro, una vez por paciente: probabilidades, vecinos y competencias."""
    n = X.n_pacientes_prueba()
    if numero is None or not 1 <= int(numero) <= n:
        aviso_rango = aviso(f"Escribe un número entre 1 y {n}.", tipo="pendiente")
        return no_update, aviso_rango, no_update, no_update
    indice = int(numero) - 1
    datos = X.caso(indice)
    datos = {**datos, "p_artefacto": float(D.predicciones_prueba()[C.MODELO_ORIGINAL].iloc[indice])
             if D.predicciones_prueba() is not None else None}
    real = C.CLASES[datos["y_real"]]
    ficha = html.Div([
        html.Dl([html.Div([html.Dt(k), html.Dd(v)]) for k, v in X.ficha(indice)], className="ficha-lista"),
        html.P([icono("person", "me-1"), "Desenlace real: ", html.Strong(real)], className="mb-0"),
    ])
    meta = D.metadata(C.MODELO_ORIGINAL)
    umbral = float(meta["umbral"])
    ps = datos["probabilidades"]
    tope = max(max(ps) * 1.25, umbral * 1.25, 0.05)
    filas = [(m, p / tope, pct(100 * p)) for m, p in zip(datos["modelos"], ps)]
    probs = _barras(filas, marca=(umbral / tope, f"Umbral operativo: {pct(100 * umbral)}"),
                    pie=f"Probabilidad de readmisión antes de 30 días de cada modelo combinado. La línea marca el "
                        f"umbral operativo ({pct(100 * umbral)}). Escala de 0 a {pct(100 * tope, 0)}.")
    return datos, ficha, probs, R.fig_proyeccion(datos, X.proyeccion())


@callback(Output("mo-lambda", "value"), Output("mo-tau", "value"), Input("mo-restablecer", "n_clicks"),
          prevent_initial_call=True)
def restablecer(_):
    h = _hiperparametros()
    return h["lam"], _indice_tau(h["tau"])


@callback(Output("mo-fig-competencias", "figure"), Output("mo-pesos", "children"), Output("mo-final", "children"),
          Output("mo-lambda-valor", "children"), Output("mo-tau-valor", "children"),
          Output("mo-lectura-decide", "children"),
          Input("mo-caso", "data"), Input("mo-lambda", "value"), Input("mo-tau", "value"))
def recombinar(datos, lam, indice_tau):
    """Lo barato, en cada movimiento: C = (1-λ)·local + λ·global → pesos → probabilidad."""
    if not datos or lam is None or indice_tau is None:
        return no_update, no_update, no_update, no_update, no_update, no_update
    h = _hiperparametros()
    # El valor del artefacto se usa tal cual (no el de la lista) para reproducirlo exacto.
    tau = h["tau"] if int(indice_tau) == _indice_tau(h["tau"]) else float(C.TAUS[int(indice_tau)])
    lam = float(lam)
    r = X.combinacion(datos, lam, tau)
    pesos = r["pesos"]
    mejor = int(np.argmax(pesos))
    es_artefacto = abs(lam - h["lam"]) < 1e-9 and tau == h["tau"]
    umbral = float(D.metadata(C.MODELO_ORIGINAL)["umbral"])
    barras = _barras([(m, w, pct(100 * w)) for m, w in zip(datos["modelos"], pesos)], destacar=mejor)
    p, p_art = r["p"], datos.get("p_artefacto")
    if es_artefacto and p_art is not None:
        comprobacion = f"Con los valores del artefacto coincide con su predicción ({pct(100 * p_art, 2)}; diferencia {num(abs(p - p_art), 6)})."
    elif p_art is not None:
        comprobacion = f"El artefacto (λ = {num(h['lam'], 1)}, τ = {num(h['tau'], 4)}) da {pct(100 * p_art, 2)}."
    else:
        comprobacion = ""
    decision = "priorizar seguimiento" if p >= umbral else "seguimiento habitual"
    final = html.Div([
        html.Div([html.Span("Probabilidad final", className="barra-nombre"),
                  html.Span(pct(100 * p, 2), className="final-numero")], className="final-fila"),
        html.Div(f"Con el umbral operativo de {pct(100 * umbral)}: {decision}. {comprobacion}", className="tarjeta-nota"),
    ])
    c_loc, c_glob = np.asarray(datos["competencia_local"]), np.asarray(datos["competencia_global"])
    mejor_local, mejor_global = datos["modelos"][int(np.argmin(c_loc))], datos["modelos"][int(np.argmin(c_glob))]
    vecinos = np.asarray(datos["vecinos_y"])
    texto = [
        f"Entre sus {len(vecinos)} vecinos hay {int(vecinos.sum())} readmitidos; la prevalencia local estimada es "
        f"{pct(100 * datos['peso_positivos'])}.",
        f"Por competencia local el mejor es {mejor_local}; por la global, {mejor_global}. Con λ = {num(lam, 2)} la "
        f"mezcla da el mayor peso a {datos['modelos'][mejor]} ({pct(100 * pesos[mejor])}).",
        (f"El mayor peso queda en {datos['modelos'][mejor]} aunque τ sea pequeño: las diferencias de Brier entre "
         "modelos son de milésimas." if max(pesos) > 0.5 else
         "Los pesos están repartidos: con este τ ningún modelo domina."),
    ]
    return (R.fig_competencias(datos, r["competencia"]), barras, final, f"= {num(lam, 2)}", f"= {num(tau, 4)}",
            lectura(texto, titulo="Lectura de este paciente"))


# =========================================================================== #
# 3 · El resultado
# =========================================================================== #
def _datos_principales():
    ev, sota = D.evaluacion_60(), D.benchmarks_sota()
    if ev is None:
        return None, None, None
    return ev, sota, E.matriz_por_fold(ev, sota)


def paso_resultado():
    ev, sota, matriz = _datos_principales()
    if ev is None:
        return html.Div([_titulo("El resultado: DES-CPL empata con los mejores"),
                         _sin_datos("el resultado por fold", "results/des_cpl_evaluacion_60.csv")])
    largo = A.por_fold_igual_presupuesto(ev, sota)
    dif = E.diferencias_con_intervalo(matriz)
    x = dif[dif["clave"] == "xgboost"]
    titulo = ("El resultado: no supera a los mejores; la diferencia con XGBoost es de "
              f"{num(x.iloc[0]['diferencia'], 4)}, sin significancia" if len(x) else "El resultado: DES-CPL empata con los mejores")
    tabla_lideres = A.tabla_igual_presupuesto(ev, sota)
    pruebas = E.pruebas_protocolo()
    adicionales = D.leer_csv_opcional(C.RESULTADOS / "adicionales" / "evaluacion_conjuntos.csv")
    condiciones = E.condiciones_superioridad(matriz, pruebas, adicionales)
    filas_dif = [[f["benchmark"], num(f["diferencia"], 4, signo=True),
                  f"{num(f['ic_bajo'], 4, signo=True)} a {num(f['ic_alto'], 4, signo=True)}",
                  f"{int(f['gana_folds'])} de {int(f['n_folds'])}", num(f["p"], 3), num(f["p_holm"], 3)]
                 for _, f in dif.sort_values("diferencia", ascending=False).iterrows()]
    return html.Div([
        _titulo(titulo),
        dbc.Row([
            dbc.Col(tarjeta_grafico(R.fig_por_fold(largo), "AUC-PR por fold de cada método",
                                    "Cada círculo es un fold externo; el punto, la media; la línea, ± 1 desviación. "
                                    "Todos con 60 evaluaciones de búsqueda."), lg=8),
            dbc.Col(lectura(A.lectura_lideres_presupuesto(tabla_lideres)), lg=4),
        ], className="g-4"),
        dbc.Row([
            dbc.Col(tarjeta_grafico(R.fig_diferencias(dif), "DES-CPL menos cada benchmark, con su intervalo del 95 %",
                                    "Varianza de Nadeau-Bengio (factor 1/5 + 1/4) y t con 4 grados de libertad. "
                                    "Franja gris: equivalencia de ±0,005 de AUC-PR."), lg=8),
            dbc.Col(lectura(E.lectura_diferencias(dif)), lg=4),
        ], className="g-4 mt-1"),
        html.Div(className="mt-3"),
        tarjeta(_tabla_corta(["Benchmark", "Diferencia", "IC del 95 %", "Gana en folds", "p (unilateral)",
                              "p con Holm (9)"], filas_dif),
                titulo="Las nueve comparaciones",
                nota="p unilateral de la t de Nadeau-Bengio (H1: DES-CPL mejor); Holm sobre los 9 benchmarks."),
        html.Div(className="mt-3"),
        dbc.Row([
            dbc.Col(_tarjeta_pruebas(pruebas), lg=7),
            dbc.Col(_tarjeta_regla(condiciones), lg=5),
        ], className="g-4"),
    ])


def _tarjeta_pruebas(pruebas):
    if pruebas is None:
        return _sin_datos("las tres pruebas del protocolo", "results/pruebas_hipotesis_modelo_original.csv")
    columnas = [c for c in ["prueba", "estadístico", "p crudo", "p ajustado (Holm)", "tamaño del efecto",
                            "conclusión"] if c in pruebas.columns]
    filas = [[E.legible(f[c]) if c not in ("p crudo", "p ajustado (Holm)")
              else p_valor(float(f[c])) for c in columnas] for _, f in pruebas.iterrows()]
    nombres = {"prueba": "Prueba", "estadístico": "Estadístico", "p crudo": "p crudo", "p ajustado (Holm)": "p con Holm",
               "tamaño del efecto": "Tamaño del efecto", "conclusión": "Conclusión"}
    return tarjeta(_tabla_corta([nombres[c] for c in columnas], filas), titulo="Las tres pruebas del protocolo",
                   nota="Fijadas antes de ver los resultados (etiqueta protocolo-v1). Comparan a DES-CPL con el "
                        "mejor benchmark, XGBoost.")


def _tarjeta_regla(condiciones):
    items = []
    for i, c in enumerate(condiciones, 1):
        estado = c["cumple"]
        simbolo, clase = ("check", "regla-si") if estado else (("remove", "regla-no") if estado is False else ("help", "regla-nd"))
        items.append(html.Li([
            html.Span(simbolo, className="material-symbols-rounded regla-icono", **{"aria-hidden": "true"}),
            html.Div([html.Div(f"{i}. {c['texto']}", className="regla-texto"),
                      html.Div([html.Strong("No se cumple. " if estado is False else
                                            "Se cumple. " if estado else "Sin datos. "), c["valor"]],
                               className="regla-valor")]),
        ], className=f"regla {clase}"))
    cumplidas = sum(1 for c in condiciones if c["cumple"])
    return tarjeta(html.Ul(items, className="reglas"), titulo="Regla de superioridad: cuatro condiciones a la vez",
                   nota=f"Se cumplen {cumplidas} de {len(condiciones)}: no hay superioridad estadística. "
                        "Protocolo, sección 5.")


# =========================================================================== #
# 4 · Por qué empata
# =========================================================================== #
def paso_empata():
    ev = D.evaluacion_60()
    if ev is None:
        return html.Div([_titulo("Por qué empata: los datos no dicen qué modelo acierta con cada paciente"),
                         _sin_datos("el análisis de λ y la ablación", "results/des_cpl_evaluacion_60.csv")])
    lambdas = E.lambda_por_fold(ev)
    abl = E.ablacion(ev)
    r2 = E.diagnostico_r2()
    unos = int((lambdas >= 1).sum())
    lectura_lambda = [
        f"La búsqueda eligió λ = 1 en {unos} de {len(lambdas)} folds y {', '.join(num(v, 1) for v in lambdas[lambdas < 1])} "
        "en los otros: casi siempre prefiere ignorar la región del paciente.",
        "λ = 1 equivale a los pesos globales, que son un ensamble estático. Si la región tuviera información "
        "útil, la búsqueda habría elegido un λ bajo.",
    ]
    mayor = abl.iloc[0]
    dentro = int((abl["diferencia"].abs() <= abl["sd"]).sum())
    lectura_abl = [
        f"Ninguna variante se separa de DES-CPL completo: {dentro} de {len(abl)} diferencias medias son menores que su "
        f"desviación entre folds. La mayor es {num(mayor['diferencia'], 4, signo=True)} "
        f"({mayor['variante'].replace('Ablación: ', '').replace('DES-CPL inicial', 'versión inicial').lower()}).",
        "Quitar la parte local (λ = 0 se parece a la ablación sin encogimiento) no ayuda; encoger hacia lo global, tampoco "
        "cambia el resultado de forma apreciable.",
    ]
    columnas_r2 = []
    if r2:
        columnas_r2 = [
            tarjeta(html.Div([html.Div(f"≤ {num(r2['maximo'], 3)}", className="kpi-gigante"),
                              html.P("R² fuera de pliegue al predecir qué modelo acierta con cada paciente, a partir "
                                     "de su perfil y de 38 variables. No hay señal que una selección local pueda "
                                     "aprovechar.", className="cuerpo mb-0")]),
                    titulo="El diagnóstico", tono="acento")]
    return html.Div([
        _titulo("Empata porque los datos no dicen qué modelo acierta con cada paciente"),
        dbc.Row([
            dbc.Col(tarjeta_grafico(R.fig_lambda_fold(lambdas), "λ elegido en cada fold: el termómetro de la parte local",
                                    "0: selección dinámica pura. 1: pesos globales."), lg=6),
            dbc.Col(tarjeta_grafico(R.fig_ablacion(abl), "Ablación: qué cambia al quitar cada componente",
                                    "Diferencia de AUC-PR de cada variante con DES-CPL completo, en los mismos 5 folds."),
                    lg=6),
        ], className="g-4"),
        dbc.Row([
            dbc.Col(lectura(lectura_lambda, titulo="Lectura del λ"), lg=6),
            dbc.Col(lectura(lectura_abl, titulo="Lectura de la ablación"), lg=6),
        ], className="g-4 mt-1"),
        html.Div(className="mt-3"),
        *columnas_r2,
    ])


# =========================================================================== #
# 5 · Generalización y estabilidad
# =========================================================================== #
def paso_generaliza():
    ev, sota = D.evaluacion_60(), D.benchmarks_sota()
    adicionales = D.leer_csv_opcional(C.RESULTADOS / "adicionales" / "evaluacion_conjuntos.csv")
    bloques = []
    titulo = "Generalización y estabilidad: se mantiene cerca de la cabeza, como los ensambles estáticos"
    if ev is None or adicionales is None:
        bloques.append(_sin_datos("el rango medio entre conjuntos",
                                  "results/des_cpl_evaluacion_60.csv y results/adicionales/evaluacion_conjuntos.csv"))
    else:
        rangos = E.rangos_en_conjuntos(ev, sota, adicionales)
        fila = rangos[rangos["clave"] == C.MODELO_ORIGINAL].iloc[0]
        puesto = int(rangos.index[rangos["clave"] == C.MODELO_ORIGINAL][0]) + 1
        lgbm = rangos[rangos["clave"] == "lightgbm"]
        individuales = rangos[~rangos["metodo"].isin(R.ESTATICOS) & (rangos["clave"] != C.MODELO_ORIGINAL)]
        mejor_ind = individuales.iloc[0]
        titulo = (f"Rango medio {num(fila['rango_4'], 2)} de {len(rangos)} en los 4 conjuntos: delante de los modelos "
                  "individuales, detrás de los ensambles estáticos")
        lineas = [f"En los 4 conjuntos (el principal y 3 adicionales) DES-CPL queda en el puesto {puesto} de "
                  f"{len(rangos)} con rango medio {num(fila['rango_4'], 2)}, por delante de todos los modelos "
                  f"individuales (el mejor, {mejor_ind['metodo']}, tiene {num(mejor_ind['rango_4'], 2)})."]
        if len(lgbm):
            lineas.append(f"Es un orden de rango medio sobre los 4: en los 3 adicionales solos DES-CPL "
                          f"({num(fila['rango_3'], 2)}) " + ("empata con" if abs(fila["rango_3"] - lgbm.iloc[0]["rango_3"]) < 0.01
                                                             else "queda frente a") +
                          f" LightGBM ({num(lgbm.iloc[0]['rango_3'], 2)}).")
        estaticos = rangos[rangos["metodo"].isin(R.ESTATICOS - {"Promedio simple"})]
        lineas.append("Los pesos globales y el apilamiento quedan por delante de DES-CPL "
                      f"({', '.join(f'{m} {num(r, 2)}' for m, r in zip(estaticos['metodo'], estaticos['rango_4']))}): "
                      "la ventaja es de combinar modelos, no de la parte local.")
        bloques.append(dbc.Row([
            dbc.Col(tarjeta_grafico(R.fig_rangos_conjuntos(rangos), "Rango medio de AUC-PR en cuatro conjuntos de datos",
                                    "Problema principal, crédito, banco y CDC. Barra: los 4 conjuntos; rombo: solo los "
                                    "3 adicionales."), lg=8),
            dbc.Col(lectura(lineas), lg=4)], className="g-4"))
    rob = D.leer_csv_opcional(C.RESULTADOS / "robustez_auc_pr.csv")
    if rob is None:
        bloques.append(_sin_datos("la robustez por escenario", "results/robustez_auc_pr.csv"))
    else:
        largo = E.robustez_perdidas(rob)
        metodos = [m for m in ["DES-CPL", "Pesos globales (Brier)", "Apilamiento logístico", "xgboost",
                               "random_forest", "logistica"] if m in set(largo["método"])]
        razon = largo[largo["método"] == "xgboost"]["razon_con_des_cpl"]
        estaticos = largo[largo["método"].isin(["Pesos globales (Brier)", "Apilamiento logístico"])]
        lineas_rob = []
        if len(razon):
            lineas_rob.append(f"XGBoost pierde de {num(razon.min(), 1)} a {num(razon.max(), 1)} veces lo que pierde DES-CPL "
                              "ante ruido, menos prevalencia y menos datos.")
        if len(estaticos):
            lineas_rob.append(f"Los ensambles estáticos pierden prácticamente lo mismo que DES-CPL (entre "
                              f"{num(estaticos['razon_con_des_cpl'].min(), 2)} y {num(estaticos['razon_con_des_cpl'].max(), 2)} "
                              "veces): es una propiedad de combinar.")
        bloques.append(dbc.Row([
            dbc.Col(tarjeta_grafico(R.fig_robustez(largo, metodos), "Robustez por escenario",
                                    "Pérdida de AUC-PR respecto al problema original en cinco perturbaciones del "
                                    "entrenamiento."), lg=8),
            dbc.Col(lectura(lineas_rob, titulo="Lectura de la robustez"), lg=4)], className="g-4 mt-1"))
    semillas = E.semillas_resumen()
    if semillas is None:
        bloques.append(_sin_datos("las 10 semillas", "results/semillas_resumen.csv"))
    else:
        s = semillas.set_index("metodo")
        propio, xgb = s.loc[C.nombre_modelo(C.MODELO_ORIGINAL)], s.loc["XGBoost"]
        otros = s.drop(index=[C.nombre_modelo(C.MODELO_ORIGINAL), "XGBoost"], errors="ignore")
        lineas_sem = [f"Con 10 semillas, DES-CPL da {num(propio['mean'], 4)} ± {num(propio['std'], 4)} y XGBoost "
                      f"{num(xgb['mean'], 4)} ± {num(xgb['std'], 4)}: es {num(xgb['std'] / propio['std'], 1)} veces más variable."]
        if len(otros):
            lineas_sem.append("Los ensambles estáticos son igual de estables ("
                              + ", ".join(f"{m}: ± {num(f['std'], 4)}" for m, f in otros.iterrows()) + ").")
        bloques.append(dbc.Row([
            dbc.Col(tarjeta_grafico(R.fig_semillas(semillas), "Estabilidad entre 10 semillas",
                                    "Semilla de la búsqueda y de la partición interna."), lg=8),
            dbc.Col(lectura(lineas_sem, titulo="Lectura de las semillas"), lg=4)], className="g-4 mt-1"))
    return html.Div([_titulo(titulo), *bloques])


# =========================================================================== #
# 6 · Comprobaciones posteriores al protocolo
# =========================================================================== #
def paso_comprobaciones():
    filas = [[clave, nombre, decimales_es(cifra), cambia] for clave, nombre, cifra, cambia in C.COMPROBACIONES]
    cambian = sum(1 for *_, cambia in C.COMPROBACIONES if cambia != "No")
    sintesis = E.sintesis_comprobaciones()
    notas = []
    if sintesis["exploraciones"]:
        e = sintesis["exploraciones"]
        notas.append(f"Lectura automática de results/exploracion/sintesis.csv ({e['n']} filas): {e['nombre']}, AUC-PR de "
                     f"{num(e['auc_pr'], 4)} ({num(e['diferencia'], 4, signo=True)} frente a XGBoost, p = {num(e['p'], 2)}).")
    if sintesis["verificaciones"]:
        v = sintesis["verificaciones"]
        extra = (f" Las que no ({', '.join(v['no_mantienen'])}) se explican en la tabla y no cambian el veredicto."
                 if v["no_mantienen"] else "")
        notas.append(f"results/verificaciones/sintesis.csv registra {v['total']} verificaciones; {v['mantienen']} mantienen "
                     f"su conclusión.{extra}")
    return html.Div([
        _titulo(f"Trece comprobaciones posteriores al protocolo: {'ninguna cambia' if not cambian else str(cambian) + ' cambian'} el veredicto"),
        tarjeta(_tabla_corta(["", "Qué se probó", "Cifra clave", "¿Cambia el veredicto?"], filas),
                titulo="Nueve verificaciones (V) y cuatro exploraciones (E)",
                nota="Las verificaciones preguntan si el resultado es correcto; las exploraciones, si DES-CPL puede "
                     "mejorarse. Ninguna forma parte del protocolo pre-registrado (capítulos 20 y 21)."),
        lectura(["Corregir cada decisión discutible deja a DES-CPL donde estaba o lo acerca a XGBoost sin superarlo.",
                 "Las variantes más fuertes suben unas milésimas, pero los ensambles estáticos con los mismos modelos "
                 "suben lo mismo o más.",
                 "El límite está en la información disponible, no en el combinador ni en el diseño de la evaluación."],
                titulo="Lectura conjunta"),
        *[html.P(n, className="nota-fuente") for n in notas],
    ])


# =========================================================================== #
# 7 · Veredicto
# =========================================================================== #
def paso_veredicto():
    veredicto = D.archivo_original("veredicto.json")
    texto = decimales_es(veredicto["texto"]) if veredicto else (
        "DES-CPL es original y propio, y empata con el mejor benchmark sin superarlo.")
    tarjetas = [
        ("R1", "Sí", "Novedad", "No hay selección dinámica de ensambles aplicada a la readmisión en diabéticos "
                                "(5 bases de literatura, capítulo 14).", True),
        ("R2", "Sí", "Implementación propia", "Programada desde cero en src/des_cpl.py, compatible con scikit-learn.", True),
        ("R3", "No", "Superioridad verificada", "Ninguna de las cuatro condiciones del protocolo se cumple.", False),
    ]
    return html.Div([
        _titulo("Veredicto: original (R1) y propio (R2), pero no superior (R3)"),
        dbc.Row([dbc.Col(html.Div([
            html.Div(r, className="veredicto-clave"), html.Div(valor, className="veredicto-valor"),
            html.Div(nombre, className="veredicto-nombre"), html.P(detalle, className="veredicto-detalle")],
            className="veredicto" + (" veredicto-si" if cumple else " veredicto-no")), md=4)
            for r, valor, nombre, detalle, cumple in tarjetas], className="g-3"),
        html.Div(className="mt-3"),
        aviso(texto + " Es un resultado reportado, no un error: los datos no contienen la información que una "
                      "selección local necesitaría.", "En una frase", "info"),
        html.P([html.A([icono("description", "me-1"), "Ficha del modelo"], href="/modelos", className="enlace-libro"),
                "  ·  ", html.A([icono("menu_book", "me-1"), "Capítulos 13 a 21 del libro"], href=C.URL_LIBRO,
                                  target="_blank", rel="noopener noreferrer", className="enlace-libro")],
               className="mt-2"),
    ])


#: Familias contra las que se mide al modelo original (el «mejor benchmark»
#: del protocolo: 7 modelos base y los 2 del estado del arte).
_BENCHMARKS = ("Modelo base", "Estado del arte")


def _kpis():
    """Cuatro cifras de entrada, leídas de los mismos archivos que los pasos 3
    y 7. Si falta un archivo, su tarjeta no aparece."""
    tarjetas = []
    nombre = C.nombre_modelo(C.MODELO_ORIGINAL)
    lideres = A.lideres_igual_presupuesto()
    if lideres is not None and lideres["es_original"].any():
        o = lideres[lideres["es_original"]].iloc[0]
        b = lideres[lideres["familia"].isin(_BENCHMARKS)].iloc[0]
        tarjetas.append({
            "valor": num(o["auc_pr_media"], 4), "etiqueta": f"AUC-PR de {nombre} (5 folds)",
            "detalle": (f"Frente a {num(b['auc_pr_media'], 4)} de {b['modelo']}, el mejor benchmark: "
                        f"{num(o['auc_pr_media'] - b['auc_pr_media'], 4, signo=True)}."),
            "color": color_modelo(C.MODELO_ORIGINAL)})
    pruebas = E.pruebas_protocolo()
    if pruebas is not None:
        nb = pruebas[pruebas["prueba"].str.contains("Nadeau", na=False)]
        if len(nb):
            f = nb.iloc[0]
            tarjetas.append({
                "valor": p_valor(float(f["p ajustado (Holm)"])), "etiqueta": "Nadeau-Bengio con Holm",
                "detalle": (f"{E.legible(f['estadístico'])}; {p_valor(float(f['p crudo']))} sin ajustar "
                            f"(Holm sobre las {len(pruebas)} pruebas del protocolo). {E.legible(f['conclusión'])}.")})
    prueba = D.leer_csv_opcional(C.RESULTADOS / "prueba_metricas.csv")
    if prueba is not None and (prueba["método"] == "DES-CPL").any():
        o = prueba[prueba["método"] == "DES-CPL"].iloc[0]
        b = prueba[prueba["familia"].isin(_BENCHMARKS)].sort_values("auc_pr", ascending=False).iloc[0]
        tarjetas.append({
            "valor": num(o["auc_pr"], 4), "etiqueta": "AUC-PR en el conjunto de prueba",
            "detalle": f"Frente a {num(b['auc_pr'], 4)} de {C.nombre_modelo(b['método'])} en la misma prueba.",
            "color": color_modelo(C.MODELO_ORIGINAL)})
    if C.MODELO_ORIGINAL in D.modelos_disponibles():
        meta = D.metadata(C.MODELO_ORIGINAL)
        tarjetas.append({
            "valor": f"{num(meta['tiempo_inferencia_ms_por_caso'], 2)} ms", "etiqueta": "Inferencia por caso",
            "detalle": (f"Ajuste completo en {num(meta['tiempo_ajuste_s'] / 60, 1)} min; artefacto de "
                        f"{num(meta['tamano_mb'], 1)} MB.")})
    return fila_kpi(tarjetas, md=12 // len(tarjetas)) if tarjetas else None


CONSTRUCTORES = {"idea": paso_idea, "decide": paso_decide, "resultado": paso_resultado, "empata": paso_empata,
                 "generaliza": paso_generaliza, "comprobaciones": paso_comprobaciones, "veredicto": paso_veredicto}


# =========================================================================== #
# Página y navegación por pasos
# =========================================================================== #
def layout():
    return html.Div([
        encabezado("04 · Modelo original", "DES-CPL: un comité que decide por paciente",
                   "Qué hace el modelo original, cómo decide con un paciente real y qué evidencia hay de que "
                   "empata con los mejores sin superarlos."),
        _kpis(),
        dbc.Tabs([dbc.Tab(label=f"{i + 1} · {nombre}", tab_id=clave, label_class_name="paso-etiqueta",
                          active_label_class_name="paso-etiqueta-activa")
                  for i, (clave, nombre) in enumerate(PASOS)],
                 id="mo-pasos", active_tab="idea", className="pasos-relato"),
        html.Div(paso_idea(), id="mo-contenido", className="paso-relato"),
        html.Div([
            dbc.Button([icono("arrow_back", "me-2"), "Anterior"], id="mo-anterior", className="boton-secundario",
                       n_clicks=0),
            dbc.Button(["Siguiente", icono("arrow_forward", "ms-2")], id="mo-siguiente", className="boton-principal",
                       n_clicks=0),
        ], className="botones pasos-navegacion"),
        dcc.Store(id="mo-scroll"),
    ], className="pagina pagina-relato")


@callback(Output("mo-contenido", "children"), Input("mo-pasos", "active_tab"))
def mostrar_paso(paso):
    """El contenido de un paso se construye solo cuando el paso está activo."""
    return CONSTRUCTORES.get(paso, paso_idea)()


@callback(Output("mo-pasos", "active_tab"), Input("mo-anterior", "n_clicks"), Input("mo-siguiente", "n_clicks"),
          State("mo-pasos", "active_tab"), prevent_initial_call=True)
def avanzar(_a, _s, actual):
    i = IDS.index(actual) if actual in IDS else 0
    i = max(0, i - 1) if ctx.triggered_id == "mo-anterior" else min(len(IDS) - 1, i + 1)
    return IDS[i]


@callback(Output("mo-anterior", "disabled"), Output("mo-siguiente", "disabled"), Input("mo-pasos", "active_tab"))
def limites(paso):
    """Anterior no sirve en el primer paso ni Siguiente en el último."""
    return paso == IDS[0], paso == IDS[-1]


# Al cambiar de paso, la vista vuelve al comienzo de la navegación por pasos.
clientside_callback(
    """function(paso) {
        var el = document.getElementById('mo-pasos');
        if (el && paso && paso !== 'idea') { el.scrollIntoView({block: 'start', behavior: 'instant'}); }
        return paso;
    }""",
    Output("mo-scroll", "data"), Input("mo-pasos", "active_tab"))
