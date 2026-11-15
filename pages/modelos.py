"""
Página 3 · Modelos: los benchmarks del Entregable 2 y el modelo original.

Contenido mínimo de la sección 7.3.3 de la guía: tabla de líderes ordenable
y filtrable, predicción en vivo con validación y explicación local,
diagnóstico (matriz de confusión, ROC, PR y calibración), interpretabilidad
global, comparación estadística con diagrama de diferencia crítica,
subsección del modelo original y ficha del modelo.

Vistas de decisión añadidas: simulador de capacidad del servicio (curva de
ganancia y matriz de 100 pacientes), termómetro de riesgo con conjunto
conforme, explicación local en cascada, «¿qué pasaría si...?», consenso de
todos los modelos, pacientes parecidos, curva de decisión, puestos por fold y
auditoría de equidad por subgrupos.

El modelo original (DES-CPL, clave ``C.MODELO_ORIGINAL``) entra por los
mismos artefactos que los benchmarks: su .joblib y su metadata en
artifacts/, su fila en mejores_combinaciones.csv y sus resultados en
artifacts/modelo_original/. Como se buscó con 60 evaluaciones y los
benchmarks del Entregable 2 con 8-12, la tabla de líderes lo marca y la
comparación estadística usa la matriz del protocolo (todos a 60).
"""

from __future__ import annotations

import re

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import ALL, Input, Output, State, callback, ctx, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.estadistica import diferencia_critica
from src.tablero import analitica as A
from src.tablero import config as C
from src.tablero import datos as D
from src.tablero import graficos as G
from src.tablero import graficos_clinicos as GC
from src.tablero import inferencia as I
from src.tablero.componentes import (aviso, encabezado, falta_artefacto, icono, kpi,
                                     glucometro, lectura, lista, matriz_personas, pendiente, semaforo,
                                     subpestanas, subtitulo, tabla, tarjeta, tarjeta_grafico, tex)
from src.tablero.formato import decimales_es, miles, num, pct
from src.tablero.tema import CLASE, COLOR_MARCA

dash.register_page(__name__, path="/modelos", name="Modelos", order=2,
                   title="Modelos · Readmisión temprana")

SUBPESTANAS = "subpestanas-modelos"


def _selector(id_, valor=None):
    modelos = D.modelos_disponibles()
    return dcc.Dropdown(id=id_, options=[{"label": C.MODELOS.get(m, m), "value": m} for m in modelos],
                        value=valor or (modelos[0] if modelos else None), clearable=False)


# =========================================================================== #
# Tabla de líderes
# =========================================================================== #
def _lideres():
    if not D.artefactos_disponibles():
        return falta_artefacto("la tabla de mejores combinaciones (artifacts/mejores_combinaciones.csv)")
    m = D.mejores().copy()
    m["Modelo"] = m["modelo"].map(lambda v: C.MODELOS.get(v, v))
    m["Balanceo"] = m["balanceo"].map(lambda v: C.BALANCEOS.get(v, v))
    m["Optimizador"] = m["optimizador"].map(lambda v: C.OPTIMIZADORES.get(v, v))
    m["AUC-PR (media ± sd)"] = [f"{num(a, 4)} ± {num(b, 4)}" for a, b in zip(m["auc_pr_media"], m["auc_pr_sd"])]
    m["AUC-ROC (media ± sd)"] = [f"{num(a, 4)} ± {num(b, 4)}" for a, b in zip(m["auc_roc_media"], m["auc_roc_sd"])]
    m["Inferencia (ms por caso)"] = [D.metadata(x)["tiempo_inferencia_ms_por_caso"]
                                     if x in D.modelos_disponibles() else None for x in m["modelo"]]
    m["Finalista"] = np.where(m["finalista"], "Sí", "No")
    es_original = m["modelo"] == C.MODELO_ORIGINAL
    m["Presupuesto"] = np.where(es_original, "60 evaluaciones (protocolo)", "Entregable 2")
    numerica = lambda d: {"type": "numeric", "format": Format(precision=d, scheme=Scheme.fixed)}
    columnas = ([{"name": c, "id": c} for c in ["Modelo", "Balanceo", "Optimizador", "AUC-PR (media ± sd)"]]
                + [{"name": "AUC-ROC (media ± sd)", "id": "AUC-ROC (media ± sd)"},
                   {"name": "Brier", "id": "brier_media", **numerica(4)},
                   {"name": "Inferencia (ms por caso)", "id": "Inferencia (ms por caso)", **numerica(3)},
                   {"name": "Rango medio", "id": "rango_medio", **numerica(2)},
                   {"name": "Finalista", "id": "Finalista"},
                   {"name": "Presupuesto", "id": "Presupuesto"}])
    benchmarks = m[~es_original]
    mejor = benchmarks.loc[benchmarks["auc_pr_media"].idxmax()]
    nota_original = []
    if es_original.any():
        original = m[es_original].iloc[0]
        protocolo = D.por_fold_protocolo()
        texto = (f"{original['Modelo']} se buscó con 60 evaluaciones y los demás con las del Entregable 2 "
                 "(8 a 12): su fila no se compara de igual a igual con el resto.")
        if protocolo is not None:
            medias = protocolo.mean().drop(C.MODELO_ORIGINAL)
            rival = medias.idxmax()
            texto += (f" Con el mismo presupuesto, {C.nombre_modelo(rival)} obtiene "
                      f"{num(medias[rival], 4)} frente a {num(protocolo[C.MODELO_ORIGINAL].mean(), 4)} de "
                      f"{C.nombre_modelo(C.MODELO_ORIGINAL)} (ver «Comparación estadística»).")
        nota_original = [texto]
    finalistas = benchmarks[benchmarks["finalista"]]["Modelo"].tolist()
    prevalencia = float(D.entrenamiento()[G.Y].mean())
    return html.Div([
        html.Div(dash_table.DataTable(
            data=m.to_dict("records"), columns=columnas, sort_action="native",
            filter_action="native", filter_options={"placeholder_text": "Filtrar..."},
            locale_format={"decimal": ",", "group": "."},
            style_as_list_view=True, style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"},
            style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "right"}
                                    for c in ["brier_media", "Inferencia (ms por caso)", "rango_medio"]],
            tooltip_data=[{"Modelo": {"value": r["motivo"], "type": "text"}} for r in m.to_dict("records")],
            tooltip_delay=200, id="tabla-lideres"), className="tabla-datos"),
        html.P("Pasa el ratón sobre un modelo para ver por qué se eligió esa combinación. La métrica es "
               "la del bucle externo de la validación anidada (5 folds agrupados por paciente).",
               className="nota-fuente"),
        dbc.Row([
            dbc.Col(tarjeta_grafico(G.fig_lideres(D.mejores(), prevalencia),
                                    "AUC-PR de la mejor combinación de cada modelo",
                                    "Oscuro: finalistas que la prueba de Nemenyi no distingue del mejor. "
                                    "La línea punteada es el AUC-PR del azar, igual a la prevalencia. "
                                    "El modelo original usa otro presupuesto de búsqueda (ver la nota)."), lg=8),
            dbc.Col(lectura([
                f"La mejor combinación del Entregable 2 es {mejor['Modelo']} ({mejor['Balanceo']} + "
                f"{mejor['Optimizador']}): AUC-PR de {num(mejor['auc_pr_media'], 4)}, "
                f"{num(mejor['auc_pr_media'] / prevalencia, 1)} veces el del azar ({num(prevalencia, 3)}).",
                f"Finalistas no distinguibles del mejor: {', '.join(finalistas)}.",
                "La desviación entre folds es mayor que la distancia entre los primeros puestos: "
                "el orden del extremo superior no es concluyente (capítulo 11).",
                *nota_original,
            ]), lg=4),
        ], className="g-4 mt-1"),
    ])


# =========================================================================== #
# Capacidad del servicio (simulador)
# =========================================================================== #
def _capacidad():
    if D.predicciones_prueba() is None:
        return falta_artefacto("las predicciones de prueba (artifacts/predicciones_prueba.parquet)")
    return html.Div([
        aviso("El hospital no puede hacer seguimiento reforzado a todos los pacientes que da de alta. Elige "
              "a qué parte puede llegar el equipo y mira cuántas readmisiones alcanzaría a prevenir cada modelo "
              "si sigue a los pacientes de mayor riesgo. Todo se calcula sobre el conjunto de prueba.",
              "¿A cuántos pacientes puede seguir el equipo?", "info"),
        dbc.Row([
            dbc.Col([html.Label("Modelo", htmlFor="cap-modelo"), _selector("cap-modelo")], md=4),
            dbc.Col([html.Label("Pacientes que el servicio puede seguir (% de mayor riesgo)",
                                htmlFor="cap-porcentaje"),
                     dcc.Slider(id="cap-porcentaje", min=5, max=30, step=1, value=15,
                                marks={v: f"{v} %" for v in range(5, 31, 5)},
                                tooltip={"placement": "bottom", "always_visible": False})],
                    md=8, className="control-simulador"),
        ], className="g-3 mb-3"),
        dcc.Loading(html.Div(id="cap-contenido"), type="dot", delay_show=300),
    ])


def _kpi_mini(valor, etiqueta):
    return html.Div([html.Div(valor, className="kpi-mini-valor"),
                     html.Div(etiqueta, className="kpi-mini-etiqueta")], className="kpi-mini")


@callback(Output("cap-contenido", "children"), Input(SUBPESTANAS, "active_tab"),
          Input("cap-modelo", "value"), Input("cap-porcentaje", "value"))
def capacidad(pestana, modelo, porcentaje):
    if pestana != "mod-capacidad" or not modelo:
        return no_update
    porcentaje = int(porcentaje or 15)
    c = A.capacidad(modelo, porcentaje)
    cien = A.a_cien({"vp": c["vp"], "fn": c["fn"], "fp": c["fp"], "vn": c["vn"]})
    curvas = {m: A.curva_capacidad(m) for m in D.modelos_disponibles()}
    mejor = max(curvas, key=lambda m: curvas[m].set_index("porcentaje").loc[porcentaje, "deteccion"])
    matriz = matriz_personas([
        {"n": cien["vp"], "clase": "p-readmitido", "etiqueta": "readmisiones detectadas a tiempo"},
        {"n": cien["fn"], "clase": "p-perdido", "relleno": False, "etiqueta": "readmisiones que se escapan"},
        {"n": cien["fp"], "clase": "p-falsa", "etiqueta": "seguidos que no habrían vuelto"},
        {"n": cien["vn"], "clase": "p-neutro", "relleno": False, "etiqueta": "ni seguidos ni readmitidos"},
    ], f"De cada 100 pacientes dados de alta: {cien['vp']} readmisiones detectadas, {cien['fn']} que se "
       f"escapan, {cien['fp']} seguimientos a quien no habría vuelto y {cien['vn']} sin seguimiento ni "
       "readmisión.")
    por_cada = "–" if not np.isfinite(c["por_readmision"]) else num(c["por_readmision"], 1)
    return html.Div([
        dbc.Row([
            dbc.Col(tarjeta([
                html.P(f"Siguiendo al {porcentaje} % de mayor riesgo, de cada 100 pacientes:",
                       className="cuerpo mb-2"),
                matriz,
            ], titulo="Cien pacientes dados de alta", clase="tarjeta-personas"), lg=5),
            dbc.Col([
                dbc.Row([
                    dbc.Col(_kpi_mini(pct(c["deteccion"]), "de las readmisiones detectadas"), xs=6, md=3),
                    dbc.Col(_kpi_mini(pct(c["precision"]), "de los seguidos reingresa (precisión)"), xs=6, md=3),
                    dbc.Col(_kpi_mini(por_cada, "seguimientos por cada readmisión detectada"), xs=6, md=3),
                    dbc.Col(_kpi_mini(miles(c["vp"] + c["fp"]), f"pacientes seguidos de {miles(c['n'])}"),
                            xs=6, md=3),
                ], className="g-2 mb-3"),
                tarjeta_grafico(GC.fig_capacidad(curvas, modelo, porcentaje),
                                "Curva de ganancia: seguidos frente a readmisiones detectadas",
                                "Cada curva es un modelo (el elegido, en oscuro); la diagonal es elegir pacientes "
                                "al azar. Cuanto más arriba, más readmisiones alcanza el mismo esfuerzo."),
            ], lg=7),
        ], className="g-4"),
        html.Div(className="mt-3"),
        lectura([
            f"Con capacidad para el {porcentaje} % de los pacientes, {C.MODELOS.get(modelo, modelo)} detecta el "
            f"{pct(c['deteccion'])} de las readmisiones; elegir al azar detectaría solo el {pct(porcentaje)}.",
            f"Hay que seguir a {por_cada} pacientes por cada readmisión detectada: es el costo de la baja "
            "señal de los datos (AUC-ROC ≈ 0,67).",
            f"Con esta capacidad, el modelo que más detecta es {C.MODELOS.get(mejor, mejor)} "
            f"({pct(curvas[mejor].set_index('porcentaje').loc[porcentaje, 'deteccion'])}).",
        ], titulo="Qué significa"),
    ])


# =========================================================================== #
# Predicción en vivo
# =========================================================================== #
def _campo(columna, regla):
    etiqueta = html.Label(C.nombre(columna), htmlFor=f"campo-{columna}", title=C.descripcion(columna))
    id_ = {"type": "campo", "index": columna}
    if columna == "age_ord":
        edades = D.etiquetas_edad()
        control = dcc.Dropdown(id=id_, clearable=False, value=int(regla["mediana"]),
                               options=[{"label": edades[k], "value": k}
                                        for k in range(int(regla["min"]), int(regla["max"]) + 1)])
    elif columna in D.roles()["binarias"]:
        control = dcc.Dropdown(id=id_, clearable=False, value=int(regla["mediana"]),
                               options=[{"label": "No", "value": 0}, {"label": "Sí", "value": 1}])
    elif regla["tipo"] == "numerica":
        control = dbc.Input(id=id_, type="number", min=regla["min"], max=regla["max"],
                            step=1 if regla["entero"] else "any", value=regla["mediana"])
        etiqueta = html.Label([C.nombre(columna),
                               html.Span(f" ({regla['min']:g} a {regla['max']:g})", className="rango")],
                              title=C.descripcion(columna))
    else:
        control = dcc.Dropdown(id=id_, clearable=False, value=regla["moda"],
                               options=[{"label": C.nivel(columna, n), "value": n} for n in regla["niveles"]])
    return dbc.Col([etiqueta, control], md=6, lg=4, className="campo")


def _formulario():
    reglas, orden = I.esquema()
    usadas, secciones = set(), []
    for titulo, columnas in C.SECCIONES_FORMULARIO:
        presentes = [c for c in columnas if c in reglas]
        usadas.update(presentes)
        if presentes:
            secciones.append((titulo, presentes))
    resto = [c for c in orden if c not in usadas and c not in I.DERIVADAS]
    if resto:
        secciones.append(("Otras variables", resto))
    return dbc.Accordion([
        dbc.AccordionItem(dbc.Row([_campo(c, reglas[c]) for c in columnas], className="g-3"),
                          title=titulo, item_id=f"sec-{i}")
        for i, (titulo, columnas) in enumerate(secciones)
    ], active_item=["sec-0", "sec-1", "sec-2"], always_open=True, className="formulario")


def _prediccion():
    if not D.modelos_disponibles():
        return falta_artefacto("los modelos serializados (artifacts/*.joblib)")
    return dbc.Row([
        dbc.Col([
            aviso("Los campos empiezan en el valor típico del entrenamiento (mediana o categoría más "
                  "frecuente). Cambia los que quieras o carga un paciente real del conjunto de prueba. "
                  "La hospitalización previa y el número de antidiabéticos activos se calculan solos "
                  "a partir de los demás campos, con la misma regla del ETL.",
                  tipo="info"),
            _formulario(),
        ], lg=7),
        dbc.Col(html.Div([
            tarjeta([
                html.Label("Modelo", htmlFor="pred-modelo"), _selector("pred-modelo"),
                html.Div([
                    dbc.Button([icono("calculate", "me-2"), "Predecir"], id="btn-predecir",
                               className="boton-principal", n_clicks=0),
                    dbc.Button([icono("shuffle", "me-2"), "Paciente de prueba al azar"], id="btn-azar",
                               className="boton-secundario", n_clicks=0),
                ], className="botones"),
                html.Div(id="pred-real", className="pred-real", **{"aria-live": "polite"}),
            ], titulo="Predicción", tono="panel"),
            dcc.Loading(html.Div(id="pred-resultado", **{"aria-live": "polite"}), type="dot", delay_show=300),
        ], className="columna-fija"), lg=5),
        # El caso validado y el modelo de la última predicción: lo leen el
        # «¿qué pasaría si...?» y el consenso sin volver a validar el formulario.
        dcc.Store(id="pred-caso"),
        dbc.Col(html.Div(id="pred-detalle"), lg=12),
    ], className="g-4")


@callback(Output({"type": "campo", "index": ALL}, "value"), Output("pred-real", "children"),
          Input("btn-azar", "n_clicks"), prevent_initial_call=True)
def paciente_al_azar(n):
    prueba = D.prueba()
    fila = prueba.sample(1, random_state=int(n)).iloc[0]
    columnas = [o["id"]["index"] for o in ctx.outputs_list[0]]
    valores = []
    for c in columnas:
        v = fila[c]
        valores.append(v.item() if hasattr(v, "item") else str(v))
    real = int(fila[G.Y])
    texto = html.Span([icono("person", "me-1"), "Paciente de prueba cargado. Desenlace real: ",
                       html.Strong(C.CLASES[real]), ". Si cambias un campo, deja de ser ese paciente."])
    return valores, texto


def _semaforo_conforme(modelo, p):
    """Conjunto de predicción conforme del caso, dicho como un semáforo."""
    info = A.conforme(modelo)
    c0, c1 = A.conjunto_caso(modelo, p)
    cobertura = (f"Cobertura nominal del 90 %; verificada en la otra mitad de la prueba: {pct(100 * info['cobertura'])} "
                 f"(readmitidos {pct(100 * info['cobertura_1'])}, no readmitidos {pct(100 * info['cobertura_0'])}).")
    if c1 and not c0:
        return semaforo("critico", "Riesgo alto con un 90 % de confianza",
                        "El conjunto de predicción conforme solo contiene «readmitido». " + cobertura,
                        "priority_high")
    if c0 and not c1:
        return semaforo("bien", "Riesgo bajo con un 90 % de confianza",
                        "El conjunto de predicción conforme solo contiene «no readmitido». " + cobertura,
                        "check_circle")
    if c0 and c1:
        return semaforo("aviso", "Incierto: el modelo no puede descartar ninguna de las dos",
                        f"Pasa con el {pct(100 * info['ambos'])} de los pacientes: los datos no bastan para "
                        "decidir con un 90 % de confianza. " + cobertura, "help")
    return semaforo("neutro", "Caso atípico", "Ninguna de las dos clases alcanza el nivel de confianza. "
                    + cobertura, "error")


@callback(Output("pred-resultado", "children"), Output("pred-detalle", "children"), Output("pred-caso", "data"),
          Input("btn-predecir", "n_clicks"),
          State("pred-modelo", "value"), State({"type": "campo", "index": ALL}, "value"),
          State({"type": "campo", "index": ALL}, "id"), prevent_initial_call=True)
def predecir(_, modelo, valores, ids):
    entrada = {i["index"]: v for i, v in zip(ids, valores)}
    caso, errores = I.construir_caso(entrada)
    if errores:
        return (dbc.Alert([html.Strong("Revisa estos campos:"), html.Ul([html.Li(e) for e in errores])],
                          color="warning", className="mt-3"), None, None)
    try:
        p = I.predecir(modelo, caso)
        explicacion = I.explicar(modelo, caso)
    except Exception as error:  # noqa: BLE001 - el error se muestra en la interfaz
        return dbc.Alert(f"No se pudo predecir: {error}", color="danger", className="mt-3"), None, None

    meta = D.metadata(modelo)
    umbral = meta["umbral"]
    marcado = p >= umbral
    percentil = I.percentil_en_prueba(modelo, p)
    predicciones = D.predicciones_prueba()
    contrib = explicacion["contribuciones"]
    principal = contrib.abs().idxmax()

    resumen = html.Div([
        tarjeta(glucometro(p, umbral, marcado, percentil), titulo="Lectura del modelo", clase="mt-3"),
        _semaforo_conforme(modelo, p) if predicciones is not None else None,
    ])

    parecidos = A.pacientes_parecidos(caso, k=20) if predicciones is not None else None
    bloque_parecidos = None
    if parecidos is not None:
        vuelven = int(parecidos[G.Y].sum())
        bloque_parecidos = tarjeta([
            matriz_personas([
                {"n": vuelven, "clase": "p-readmitido", "etiqueta": "fueron readmitidos"},
                {"n": len(parecidos) - vuelven, "clase": "p-neutro", "relleno": False,
                 "etiqueta": "no fueron readmitidos"},
            ], f"De los 20 pacientes de prueba más parecidos, {vuelven} fueron readmitidos.", columnas=10),
            html.P(f"De los 20 pacientes reales más parecidos a este (conjunto de prueba), {vuelven} volvieron "
                   f"antes de 30 días: {pct(100 * vuelven / len(parecidos), 0)}, frente al "
                   f"{pct(100 * D.entrenamiento()[G.Y].mean())} general.", className="cuerpo mb-1"),
            html.P("Parecido: misma combinación de edad, utilización previa, estancia, diagnóstico principal, tipo "
                   "de admisión, destino al alta, HbA1c e insulina (distancia de Gower). Con 20 casos, la "
                   "cifra es orientativa.", className="tarjeta-nota mb-0"),
        ], titulo="Pacientes parecidos: qué les pasó", clase="tarjeta-personas")

    opciones_qps = [{"label": C.nombre(v), "value": v} for v in I.QUE_PASA_SI]
    detalle = html.Div([
        dbc.Row([
            dbc.Col(tarjeta_grafico(GC.fig_shap_cascada(contrib, caso, explicacion["base"],
                                                        explicacion["prediccion"]),
                                    "Por qué predice esto: de la tasa base a este paciente",
                                    f"Valores de Shapley sobre las variables originales ({C.PERMUTACIONES_EN_VIVO} "
                                    "permutaciones, 20 pacientes de referencia). Ámbar sube el riesgo y azul lo "
                                    "baja; la barra final es la probabilidad del paciente."), lg=8),
            dbc.Col(lectura([
                f"Lo que más mueve esta predicción es {C.nombre(principal).lower()} "
                f"({num(100 * contrib[principal], 1, signo=True)} puntos).",
                "Es una explicación de lo que hace el modelo, no de las causas de la readmisión.",
            ]), lg=4),
        ], className="g-4 mt-1"),
        dbc.Row([
            dbc.Col(tarjeta([
                html.Label("Variable que se mueve", htmlFor="qps-var"),
                dcc.Dropdown(id="qps-var", options=opciones_qps, value="number_inpatient", clearable=False,
                             className="mb-2"),
                dcc.Loading(dcc.Graph(id="qps-grafico", figure=G.figura_inicial(), config={"displaylogo": False, "responsive": True}),
                            type="dot", delay_show=300),
                html.Div("El resto de los datos del paciente queda fijo. Muestra cómo responde el modelo, no qué "
                         "ocurriría si se interviniera: es asociación, no causalidad.", className="tarjeta-nota"),
            ], titulo="¿Qué pasaría si...?"), lg=7),
            dbc.Col(bloque_parecidos, lg=5),
        ], className="g-4 mt-1"),
        dbc.Row([
            dbc.Col(tarjeta([
                html.P("¿Todos los modelos ven igual a este paciente? Si coinciden, la predicción es más "
                       "fiable; si discrepan, conviene mirarla con cautela.", className="cuerpo"),
                dbc.Button([icono("groups", "me-2"), f"Consultar a los {len(D.modelos_disponibles())} modelos"], id="btn-consenso",
                           className="boton-secundario", n_clicks=0),
                dcc.Loading(html.Div(id="consenso-salida"), type="dot", delay_show=300),
            ], titulo="Consenso de los modelos"), lg=7),
            dbc.Col(tarjeta_grafico(G.fig_cohorte(predicciones[modelo], predicciones["y_real"], p, umbral),
                                    "Dónde cae entre los pacientes de prueba",
                                    "Distribución de la probabilidad predicha en prueba, separada por el "
                                    "desenlace real de cada paciente.") if predicciones is not None else None,
                    lg=5),
        ], className="g-4 mt-1"),
    ])
    return resumen, detalle, {"modelo": modelo, "entrada": entrada}


def _caso_guardado(datos):
    if not datos:
        return None, None
    caso, errores = I.construir_caso(datos["entrada"])
    return (None, None) if errores else (datos["modelo"], caso)


@callback(Output("qps-grafico", "figure"), Input("qps-var", "value"), State("pred-caso", "data"))
def que_pasa_si(variable, datos):
    modelo, caso = _caso_guardado(datos)
    if caso is None or not variable:
        return G._vacio("Haz primero una predicción.")
    valores, probabilidades = I.barrido(modelo, caso, variable)
    return GC.fig_que_pasa_si(valores, probabilidades, int(caso[variable].iloc[0]), variable,
                              D.metadata(modelo)["umbral"])


@callback(Output("consenso-salida", "children"), Input("btn-consenso", "n_clicks"),
          State("pred-caso", "data"), prevent_initial_call=True)
def consenso(_, datos):
    modelo, caso = _caso_guardado(datos)
    if caso is None:
        return aviso("Haz primero una predicción.", tipo="pendiente")
    try:
        tabla_c = I.predecir_todos(caso)
    except Exception as error:  # noqa: BLE001
        return dbc.Alert(f"No se pudo consultar a los modelos: {error}", color="danger")
    priorizan = int((tabla_c["p"] >= tabla_c["umbral"]).sum())
    total = len(tabla_c)
    rango = (tabla_c["p"].min(), tabla_c["p"].max())
    if priorizan in (0, total):
        acuerdo = (f"Los {total} modelos coinciden en la decisión ({'priorizar' if priorizan else 'seguimiento habitual'}), "
                   f"aunque sus probabilidades van de {pct(100 * rango[0])} a {pct(100 * rango[1])}.")
    else:
        acuerdo = (f"Los modelos no coinciden ({priorizan} a favor y {total - priorizan} en contra): para este "
                   f"paciente la decisión depende del modelo elegido. Probabilidades de {pct(100 * rango[0])} "
                   f"a {pct(100 * rango[1])}.")
    return html.Div([
        dcc.Graph(figure=GC.fig_consenso(tabla_c, modelo), config={"displaylogo": False, "responsive": True}),
        lectura([
            f"{priorizan} de {total} modelos priorizarían a este paciente con su propio umbral operativo.",
            acuerdo,
        ]),
    ], className="mt-3")


# =========================================================================== #
# Diagnóstico
# =========================================================================== #
def _diagnostico():
    if D.predicciones_prueba() is None:
        return falta_artefacto("las predicciones de prueba (artifacts/predicciones_prueba.parquet)")
    return html.Div([
        dbc.Row([
            dbc.Col([html.Label("Modelo"), _selector("diag-modelo")], md=4),
            dbc.Col([html.Label("Umbral"), dbc.RadioItems(
                id="diag-umbral", inline=True, value="operativo",
                options=[{"label": "Operativo (marca al 15 % de mayor riesgo)", "value": "operativo"},
                         {"label": "0,5 (por defecto)", "value": "defecto"}])], md=8),
        ], className="g-3 mb-3"),
        dcc.Loading(html.Div(id="diag-contenido"), type="dot", delay_show=300),
    ])


@callback(Output("diag-contenido", "children"), Input(SUBPESTANAS, "active_tab"),
          Input("diag-modelo", "value"), Input("diag-umbral", "value"))
def diagnostico(pestana, modelo, tipo_umbral):
    if pestana != "mod-diagnostico" or not modelo:
        return no_update
    pred = D.predicciones_prueba()
    y, p = pred["y_real"].to_numpy(), pred[modelo].to_numpy()
    meta = D.metadata(modelo)
    umbral = meta["umbral"] if tipo_umbral == "operativo" else 0.5
    marcado = p >= umbral
    recall = float((marcado & (y == 1)).sum() / max(1, (y == 1).sum()))
    precision = float((marcado & (y == 1)).sum() / max(1, marcado.sum()))
    fig_cal, ece = G.fig_calibracion(y, p)
    fig_decision, lineas_decision = GC.fig_decision(A.curva_decision(modelo), modelo, meta["umbral"])
    return html.Div([
        dbc.Row([
            dbc.Col(tarjeta_grafico(G.fig_confusion(y, p, umbral), "Matriz de confusión",
                                    f"Conjunto de prueba ({miles(len(y))} encuentros), umbral {num(umbral, 3)}. "
                                    "Porcentajes por fila (por desenlace real)."), lg=6),
            dbc.Col(lectura([
                f"Con este umbral se marca al {pct(100 * marcado.mean())} de los pacientes y se detecta el "
                f"{pct(100 * recall)} de las readmisiones; de los marcados, el {pct(100 * precision)} reingresa.",
                "Con el umbral 0,5 el modelo casi no marca a nadie: por eso el umbral se fija por la "
                "capacidad de seguimiento del servicio (capítulo 10).",
                f"Error de calibración esperado (ECE, 10 deciles): {num(ece, 4)}.",
            ]), lg=6),
        ], className="g-4"),
        dbc.Row([
            dbc.Col(tarjeta_grafico(fig_decision, "Curva de decisión: ¿vale la pena usar el modelo?",
                                    "Beneficio neto (Vickers y Elkin, 2006) del modelo frente a intervenir a "
                                    "todos o a nadie, para cada umbral de probabilidad. Es la forma estándar "
                                    "de evaluar un modelo clínico para tomar decisiones."), lg=8),
            dbc.Col(lectura(lineas_decision, titulo="Cómo leerla"), lg=4),
        ], className="g-4 mt-1"),
        dbc.Row([
            dbc.Col(tarjeta_grafico(G.fig_roc(y, p), "Curva ROC", "Conjunto de prueba."), lg=4),
            dbc.Col(tarjeta_grafico(G.fig_pr(y, p), "Curva de precisión-exhaustividad",
                                    "La línea punteada es la precisión del azar, igual a la prevalencia."), lg=4),
            dbc.Col(tarjeta_grafico(fig_cal, "Calibración",
                                    "Cada punto es un decil de probabilidad predicha frente a la frecuencia "
                                    "observada; sobre la diagonal, las probabilidades son fiables."), lg=4),
        ], className="g-4 mt-1"),
    ])


# =========================================================================== #
# Interpretabilidad global
# =========================================================================== #
def _interpretabilidad():
    if D.shap_global() is None:
        return falta_artefacto("la explicación global (artifacts/shap_global.parquet)")
    return html.Div([
        dbc.Row(dbc.Col([html.Label("Modelo"), _selector("shap-modelo")], md=4), className="mb-3"),
        dcc.Loading(html.Div(id="shap-contenido"), type="dot", delay_show=300),
    ])


@callback(Output("shap-contenido", "children"), Input(SUBPESTANAS, "active_tab"),
          Input("shap-modelo", "value"))
def interpretabilidad(pestana, modelo):
    if pestana != "mod-interpretabilidad" or not modelo:
        return no_update
    tabla_shap = D.shap_global()
    if modelo not in set(tabla_shap["modelo"]):
        return aviso("Este modelo aún no tiene explicación global calculada.", tipo="pendiente")
    figura, importancia = G.fig_shap_global(tabla_shap, modelo)
    top = [C.nombre(v).lower() for v in importancia.index[:3]]
    return dbc.Row([
        dbc.Col(tarjeta_grafico(figura, "Resumen SHAP: importancia global",
                                "Media del valor absoluto de Shapley sobre 150 pacientes de prueba, en "
                                "puntos porcentuales de probabilidad."), lg=8),
        dbc.Col(lectura([
            f"Las tres variables que más mueven las predicciones son {top[0]}, {top[1]} y {top[2]}.",
            "Importancia no es causalidad: indica en qué se apoya el modelo para ordenar el riesgo.",
            "Las importancias de variables correlacionadas (insulina, cambio de medicación y medicación "
            "antidiabética) se reparten entre ellas (capítulo 5).",
        ]), lg=4),
    ], className="g-4")


# =========================================================================== #
# Comparación estadística
# =========================================================================== #
def _comparacion():
    matriz, es_protocolo = A.matriz_comparacion()
    k, n = matriz.shape[1], matriz.shape[0]
    cd = diferencia_critica(k, n)
    chi2, p = stats.friedmanchisquare(*[matriz[c] for c in matriz.columns])
    original = C.MODELO_ORIGINAL if C.MODELO_ORIGINAL in matriz.columns else None
    rangos_medios = matriz.rank(axis=1, ascending=False).mean().sort_values()
    lider = rangos_medios.index[0]
    if original:
        distancia = rangos_medios[original] - rangos_medios[lider]
        lectura_original = (
            f"{C.nombre_modelo(original)} queda en el rango medio {num(rangos_medios[original], 2)}, a "
            f"{num(distancia, 2)} posiciones de {C.nombre_modelo(lider)}: "
            + ("menos que la diferencia crítica, así que Nemenyi no los distingue. Empata con los mejores, "
               "pero no los supera (capítulo 13.5)." if distancia <= cd else
               "más que la diferencia crítica, así que Nemenyi sí los distingue."))
    else:
        lectura_original = ("XGBoost y Random Forest no se distinguen; el estudio de convergencia del capítulo 12 "
                            "muestra que el empate no se debe a una búsqueda corta.")
    efectos = D.tamanos_efecto()
    filas_efecto = []
    if efectos is not None:
        top = efectos.reindex(efectos["d de Cohen (Hedges)"].abs().sort_values().index).head(6)
        for _, f in top.iterrows():
            filas_efecto.append([f"{C.MODELOS.get(f['modelo A'], f['modelo A'])} vs. "
                                 f"{C.MODELOS.get(f['modelo B'], f['modelo B'])}",
                                 num(f["delta de Cliff"], 2), num(f["d de Cohen (Hedges)"], 2),
                                 f["magnitud d"]])
    return html.Div([
        dbc.Row([
            dbc.Col(tarjeta_grafico(G.fig_diferencia_critica(matriz, cd, original),
                                    "Diagrama de diferencia crítica (Nemenyi)",
                                    "Rango medio de cada modelo en AUC-PR sobre los 5 folds externos. Los "
                                    "modelos unidos por una barra no se distinguen al 5 %."
                                    + (" Todos con presupuesto de 60 evaluaciones (protocolo del capítulo 13), "
                                       "incluidos LightGBM y CatBoost." if es_protocolo else "")
                                    + (" El modelo original aparece en negrita." if original else
                                       " El modelo original se añadirá cuando exista su resultado.")), lg=8),
            dbc.Col(lectura([
                f"Friedman rechaza la igualdad de los {k} modelos: $\\chi^2 = {num(chi2, 2)}$, "
                f"{'$p < 0{,}0001$' if p < 1e-4 else f'$p = {num(p, 4)}$'}.",
                f"La diferencia crítica es de {num(cd, 2)} posiciones de rango: con 5 folds, la prueba "
                "separa a los mejores de los peores, pero no ordena el extremo superior.",
                lectura_original,
            ]), lg=4),
        ], className="g-4"),
        dbc.Row([
            dbc.Col(tarjeta_grafico(GC.fig_rangos(A.rangos_por_fold(), C.MODELO_ORIGINAL), "Puesto de cada modelo en cada fold",
                                    "1 = mejor AUC-PR en ese fold externo. En color, los dos mejores por rango "
                                    "medio; en raya discontinua, el modelo original. Si las líneas se cruzan, el orden "
                                    "depende de la partición."), lg=8),
            dbc.Col(lectura(_lectura_rangos()), lg=4),
        ], className="g-4 mt-1"),
        html.Div(className="mt-3"),
        (tarjeta(tabla(["Comparación", "Delta de Cliff", "d de Cohen (g)", "Magnitud"], filas_efecto),
                 titulo="Pares más parejos: tamaño del efecto",
                 nota="Los seis pares con menor |g|. Tabla completa en results/tamanos_efecto_cohen.csv.")
         if filas_efecto else None),
    ])


def _lectura_rangos():
    rangos = A.rangos_por_fold()
    lideres = rangos.idxmin(axis=1)
    distintos = sorted(set(lideres), key=lambda m: -int((lideres == m).sum()))
    texto = ", ".join(f"{C.nombre_modelo(m)} ({int((lideres == m).sum())})" for m in distintos)
    return [f"El primer puesto cambia de un fold a otro: lo ocupan {texto}.",
            "Por eso el diagrama de diferencia crítica no separa a los primeros: con 5 folds, el orden del "
            "extremo superior no es estable."]


# =========================================================================== #
# Equidad
# =========================================================================== #
def _equidad():
    if D.predicciones_prueba() is None:
        return falta_artefacto("las predicciones de prueba (artifacts/predicciones_prueba.parquet)")
    return html.Div([
        aviso("Un modelo puede funcionar bien en promedio y peor en un grupo concreto. Aquí se mide en el "
              "conjunto de prueba, por subgrupo, con el umbral operativo de cada modelo. El grupo racial es un "
              "atributo protegido: se audita, pero ninguna decisión clínica debería basarse en él.",
              "¿Funciona igual para todos?", "alcance"),
        dbc.Row([
            dbc.Col([html.Label("Modelo", htmlFor="eq-modelo"), _selector("eq-modelo")], md=4),
            dbc.Col([html.Label("Atributo"), dbc.RadioItems(
                id="eq-atributo", inline=True, value="edad",
                options=[{"label": v, "value": k} for k, v in A.ATRIBUTOS_EQUIDAD.items()])], md=8),
        ], className="g-3 mb-3"),
        dcc.Loading(html.Div(id="eq-contenido"), type="dot", delay_show=300),
    ])


@callback(Output("eq-contenido", "children"), Input(SUBPESTANAS, "active_tab"),
          Input("eq-modelo", "value"), Input("eq-atributo", "value"))
def equidad(pestana, modelo, atributo):
    if pestana != "mod-equidad" or not modelo:
        return no_update
    tabla_eq = A.equidad(modelo, atributo)
    y, p = D.predicciones_prueba()["y_real"].to_numpy(), D.predicciones_prueba()[modelo].to_numpy()
    umbral = D.metadata(modelo)["umbral"]
    global_ = {"auc_roc": roc_auc_score(y, p), "deteccion": 100 * ((p >= umbral) & (y == 1)).sum() / y.sum(),
               "marcados": 100 * (p >= umbral).mean()}
    figura, lineas = GC.fig_equidad(tabla_eq, global_, atributo)
    filas = [[f["grupo"], miles(f["n"]), pct(f["prevalencia"]), num(f["auc_roc"], 3), num(f["auc_pr"], 3),
              pct(f["deteccion"]), pct(f["marcados"])] for _, f in tabla_eq.iterrows()]
    return html.Div([
        dbc.Row([
            dbc.Col(tarjeta_grafico(figura, f"Desempeño por {A.ATRIBUTOS_EQUIDAD[atributo].lower()}",
                                    "Cada punto es un subgrupo de la prueba; la línea punteada, el valor en toda la "
                                    "prueba."), lg=8),
            dbc.Col(lectura(lineas), lg=4),
        ], className="g-4"),
        html.Div(className="mt-3"),
        tarjeta(tabla(["Grupo", "Encuentros", "Readmisión", "AUC-ROC", "AUC-PR", "Detectadas", "Priorizados"],
                      filas), titulo="Tabla por subgrupo",
                nota="AUC-PR depende de la prevalencia de cada grupo: compárala con su columna «Readmisión»."),
    ])


# =========================================================================== #
# Modelo original y ficha
# =========================================================================== #
def _legible(texto: str) -> str:
    """Coma decimal y nombres de modelo legibles en un texto de artifacts/."""
    texto = decimales_es(texto)
    texto = re.sub(r"\b(arbol|random_forest|xgboost|svm|logistica|knn|bayes|lightgbm|catboost)\b",
                   lambda m: C.nombre_modelo(m.group(1)), texto)
    # La lista de modelos podados se escribió con comas: se cierra con «y».
    return texto.replace("Árbol de decisión, Random Forest, XGBoost, SVM lineal",
                         "árbol de decisión, Random Forest, XGBoost y SVM lineal")


def _modelo_original():
    resumen = D.archivo_original("resumen.md")
    if resumen is None:
        return html.Div([
            aviso("Esta sección se llena sola cuando el modelo original deje sus resultados en "
                  "artifacts/modelo_original/. Mientras tanto, esto es lo que mostrará:", tipo="pendiente"),
            dbc.Row([
                dbc.Col(pendiente("El modelo", [
                    "Descripción y limitación de los métodos actuales que resuelve",
                    "Pseudocódigo con entradas, salidas y criterio de parada",
                    "Diferencias frente a 5 a 10 métodos afines de la literatura",
                    "Curva de convergencia o de aprendizaje",
                ]), lg=6),
                dbc.Col(pendiente("La evidencia", [
                    "Tabla de pruebas de hipótesis con p-valores crudos y ajustados (Holm)",
                    "Tamaño del efecto frente al mejor benchmark",
                    "Resumen de los estudios de robustez y de los conjuntos adicionales",
                    "Veredicto explícito de superioridad o no superioridad (R3)",
                ]), lg=6),
            ], className="g-4"),
        ])
    pruebas = D.archivo_original("pruebas_hipotesis.json")
    veredicto = D.archivo_original("veredicto.json")
    # Los archivos de artifacts/ siguen la convención del libro (punto decimal y
    # claves de los modelos); el tablero escribe con coma y nombres legibles.
    resumen = _legible(resumen)
    if pruebas:
        pruebas = [{k: _legible(str(v)) for k, v in f.items()} for f in pruebas]
    if veredicto:
        veredicto = {**veredicto, "texto": _legible(veredicto.get("texto", ""))}
    return html.Div([
        tarjeta(dcc.Markdown(resumen, mathjax=True, className="markdown")),
        (tarjeta(tabla(list(pruebas[0].keys()), [list(f.values()) for f in pruebas]),
                 titulo="Pruebas de hipótesis") if pruebas else None),
        (aviso(veredicto.get("texto", ""), veredicto.get("titulo", "Veredicto"),
               "info" if veredicto.get("superior") else "alcance") if veredicto else None),
    ])


def _ficha():
    if not C.MODEL_CARD.exists():
        return pendiente("Ficha del modelo (MODEL_CARD.md)", [
            "Uso previsto y usuarios", "Datos de entrenamiento y su procedencia",
            "Métricas con su variabilidad", "Limitaciones y riesgos", "Usos no recomendados",
        ])
    return tarjeta(dcc.Markdown(decimales_es(C.MODEL_CARD.read_text(encoding="utf-8")), mathjax=True,
                                className="markdown"))


def layout():
    return html.Div([
        encabezado("03 · Modelos", "Modelos: comparar, predecir y explicar",
                   "La mejor combinación de cada uno de los 7 modelos del Entregable 2, entrenada con todo "
                   "el entrenamiento y evaluada en prueba, y el modelo original del Entregable 3."),
        subpestanas([
            ("mod-lideres", "Tabla de líderes", _lideres()),
            ("mod-capacidad", "Capacidad del servicio", _capacidad()),
            ("mod-prediccion", "Predicción en vivo", _prediccion()),
            ("mod-diagnostico", "Diagnóstico", _diagnostico()),
            ("mod-interpretabilidad", "Interpretabilidad", _interpretabilidad()),
            ("mod-comparacion", "Comparación estadística", _comparacion()),
            ("mod-equidad", "Equidad", _equidad()),
            ("mod-original", "Modelo original", _modelo_original()),
            ("mod-ficha", "Ficha del modelo", _ficha()),
        ], SUBPESTANAS),
    ], className="pagina")
