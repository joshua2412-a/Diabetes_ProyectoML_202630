"""
Página 3 · Modelos: los benchmarks del Entregable 2 y el modelo original.

Contenido mínimo de la sección 7.3.3 de la guía: tabla de líderes ordenable
y filtrable, predicción en vivo con validación y explicación local,
diagnóstico (matriz de confusión, ROC, PR y calibración), interpretabilidad
global, comparación estadística con diagrama de diferencia crítica,
subsección del modelo original y ficha del modelo.

El modelo original se integra sin tocar este archivo: basta con que el
script que lo entrene escriba su artefacto y su metadata en artifacts/, su
fila en mejores_combinaciones.csv, y sus resultados en
artifacts/modelo_original/ (ver CLAUDE.md, sección del dashboard).
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import ALL, Input, Output, State, callback, ctx, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme
from scipy import stats

from src.estadistica import diferencia_critica
from src.tablero import config as C
from src.tablero import datos as D
from src.tablero import graficos as G
from src.tablero import inferencia as I
from src.tablero.componentes import (aviso, encabezado, falta_artefacto, icono, kpi,
                                     lectura, lista, pendiente, subpestanas, subtitulo,
                                     tabla, tarjeta, tarjeta_grafico, tex)
from src.tablero.formato import miles, num, pct
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
    numerica = lambda d: {"type": "numeric", "format": Format(precision=d, scheme=Scheme.fixed)}
    columnas = ([{"name": c, "id": c} for c in ["Modelo", "Balanceo", "Optimizador", "AUC-PR (media ± sd)"]]
                + [{"name": "AUC-ROC (media ± sd)", "id": "AUC-ROC (media ± sd)"},
                   {"name": "Brier", "id": "brier_media", **numerica(4)},
                   {"name": "Inferencia (ms por caso)", "id": "Inferencia (ms por caso)", **numerica(3)},
                   {"name": "Rango medio", "id": "rango_medio", **numerica(2)},
                   {"name": "Finalista", "id": "Finalista"}])
    mejor = m.iloc[0]
    finalistas = m[m["finalista"]]["Modelo"].tolist()
    prevalencia = float(D.entrenamiento()[G.Y].mean())
    return html.Div([
        dash_table.DataTable(
            data=m.to_dict("records"), columns=columnas, sort_action="native",
            filter_action="native", filter_options={"placeholder_text": "Filtrar..."},
            locale_format={"decimal": ",", "group": "."},
            style_as_list_view=True, style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "inherit", "fontSize": "0.9rem", "padding": "8px", "textAlign": "left"},
            style_header={"fontWeight": 600},
            tooltip_data=[{"Modelo": {"value": r["motivo"], "type": "text"}} for r in m.to_dict("records")],
            tooltip_delay=200, id="tabla-lideres"),
        html.P("Pasa el ratón sobre un modelo para ver por qué se eligió esa combinación. La métrica es "
               "la del bucle externo de la validación anidada (5 folds agrupados por paciente).",
               className="nota-fuente"),
        dbc.Row([
            dbc.Col(tarjeta_grafico(G.fig_lideres(D.mejores(), prevalencia),
                                    "AUC-PR de la mejor combinación de cada modelo",
                                    "Oscuro: finalistas que la prueba de Nemenyi no distingue del mejor. "
                                    "La línea punteada es el AUC-PR del azar, igual a la prevalencia."), lg=8),
            dbc.Col(lectura([
                f"La mejor combinación global es {mejor['Modelo']} ({mejor['Balanceo']} + "
                f"{mejor['Optimizador']}): AUC-PR de {num(mejor['auc_pr_media'], 4)}, "
                f"{num(mejor['auc_pr_media'] / prevalencia, 1)} veces el del azar ({num(prevalencia, 3)}).",
                f"Finalistas no distinguibles del mejor: {', '.join(finalistas)}.",
                "La desviación entre folds es mayor que la distancia entre los primeros puestos: "
                "el orden del extremo superior no es concluyente (capítulo 11).",
            ]), lg=4),
        ], className="g-4 mt-1"),
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
                               options=[{"label": n, "value": n} for n in regla["niveles"]])
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


@callback(Output("pred-resultado", "children"), Input("btn-predecir", "n_clicks"),
          State("pred-modelo", "value"), State({"type": "campo", "index": ALL}, "value"),
          State({"type": "campo", "index": ALL}, "id"), prevent_initial_call=True)
def predecir(_, modelo, valores, ids):
    caso, errores = I.construir_caso({i["index"]: v for i, v in zip(ids, valores)})
    if errores:
        return dbc.Alert([html.Strong("Revisa estos campos:"), html.Ul([html.Li(e) for e in errores])],
                         color="warning", className="mt-3")
    try:
        p = I.predecir(modelo, caso)
        explicacion = I.explicar(modelo, caso)
    except Exception as error:  # noqa: BLE001 - el error se muestra en la interfaz
        return dbc.Alert(f"No se pudo predecir: {error}", color="danger", className="mt-3")

    meta = D.metadata(modelo)
    umbral = meta["umbral"]
    marcado = p >= umbral
    percentil = I.percentil_en_prueba(modelo, p)
    predicciones = D.predicciones_prueba()
    cohorte = (tarjeta_grafico(G.fig_cohorte(predicciones[modelo], predicciones["y_real"], p, umbral),
                               "Dónde cae entre los pacientes de prueba",
                               "Distribución de la probabilidad predicha en prueba, separada por el "
                               "desenlace real de cada paciente.")
               if predicciones is not None else None)
    contrib = explicacion["contribuciones"]
    principal = contrib.abs().idxmax()
    return html.Div([
        dbc.Row([
            dbc.Col(kpi(pct(100 * p), "Probabilidad de readmisión en < 30 días",
                        f"Percentil {num(percentil, 0)} entre los pacientes de prueba" if percentil is not None else "",
                        CLASE[1] if marcado else CLASE[0]), sm=6),
            dbc.Col(kpi("Priorizar" if marcado else "Seguimiento habitual", "Decisión operativa",
                        f"Umbral {num(umbral, 3)}: {meta['umbral_criterio'].lower()}",
                        CLASE[1] if marcado else CLASE[0]), sm=6),
        ], className="g-3 mt-2"),
        aviso("Incertidumbre: los modelos benchmark no producen intervalos ni conjuntos de predicción. "
              "La predicción conforme con su cobertura nominal se añade con el modelo original "
              "(evaluación ampliada del Entregable 3).", tipo="alcance"),
        tarjeta_grafico(G.fig_shap_local(contrib, caso), "Por qué predice esto: explicación local",
                        f"Valores de Shapley sobre las variables originales ({C.PERMUTACIONES_EN_VIVO} "
                        "permutaciones, 20 pacientes de referencia). Rojo empuja hacia la readmisión y azul "
                        f"la aleja. La base es {pct(100 * explicacion['base'])}; base más aportes = "
                        f"{pct(100 * explicacion['prediccion'])}."),
        html.Div(className="mt-3"),
        lectura([f"Lo que más mueve esta predicción es {C.nombre(principal).lower()} "
                 f"({num(100 * contrib[principal], 1, signo=True)} puntos).",
                 "Es una explicación de lo que hace el modelo, no de las causas de la readmisión."]),
        html.Div(className="mt-3"),
        cohorte,
    ])


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
    matriz = D.por_fold_mejores()
    k, n = matriz.shape[1], matriz.shape[0]
    cd = diferencia_critica(k, n)
    chi2, p = stats.friedmanchisquare(*[matriz[c] for c in matriz.columns])
    original = next((c for c in matriz.columns if c not in C.MODELOS), None)
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
                                    + (" El modelo original aparece en negrita." if original else
                                       " El modelo original se añadirá cuando exista su resultado.")), lg=8),
            dbc.Col(lectura([
                f"Friedman rechaza la igualdad de los {k} modelos: $\\chi^2 = {num(chi2, 2)}$, "
                f"{'$p < 0{,}0001$' if p < 1e-4 else f'$p = {num(p, 4)}$'}.",
                f"La diferencia crítica es de {num(cd, 2)} posiciones de rango: con 5 folds, la prueba "
                "separa a los mejores de los peores, pero no ordena el extremo superior.",
                "XGBoost y Random Forest no se distinguen; el estudio de convergencia del capítulo 12 "
                "muestra que el empate no se debe a una búsqueda corta.",
            ]), lg=4),
        ], className="g-4"),
        (tarjeta(tabla(["Comparación", "Delta de Cliff", "d de Cohen (g)", "Magnitud"], filas_efecto),
                 titulo="Pares más parejos: tamaño del efecto",
                 nota="Los seis pares con menor |g|. Tabla completa en results/tamanos_efecto_cohen.csv.")
         if filas_efecto else None),
    ])


# =========================================================================== #
# Modelo original y ficha
# =========================================================================== #
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
    return tarjeta(dcc.Markdown(C.MODEL_CARD.read_text(encoding="utf-8"), mathjax=True,
                                className="markdown"))


def layout():
    return html.Div([
        encabezado("03 · Modelos", "Modelos: comparar, predecir y explicar",
                   "La mejor combinación de cada uno de los 7 modelos del Entregable 2, entrenada con todo "
                   "el entrenamiento y evaluada en prueba, y el modelo original del Entregable 3."),
        subpestanas([
            ("mod-lideres", "Tabla de líderes", _lideres()),
            ("mod-prediccion", "Predicción en vivo", _prediccion()),
            ("mod-diagnostico", "Diagnóstico", _diagnostico()),
            ("mod-interpretabilidad", "Interpretabilidad", _interpretabilidad()),
            ("mod-comparacion", "Comparación estadística", _comparacion()),
            ("mod-original", "Modelo original", _modelo_original()),
            ("mod-ficha", "Ficha del modelo", _ficha()),
        ], SUBPESTANAS),
    ], className="pagina")
