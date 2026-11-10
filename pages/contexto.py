"""
Página 1 · Contexto: el problema, los datos, la metodología y el alcance.

Contenido mínimo de la sección 7.3.1 de la guía: problema y relevancia con
una cifra citada, pregunta de investigación y objetivos, descripción de los
datos con su diccionario, diagrama del pipeline, y limitaciones, ética,
autores, versión y fecha.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, html

from src.tablero import config as C
from src.tablero import datos as D
from src.tablero.componentes import (aviso, encabezado, fila_kpi, icono, lista,
                                     parrafo, subpestanas, subtitulo, tabla, tarjeta)
from src.tablero.formato import miles, pct
from src.tablero.tema import CLASE

dash.register_page(__name__, path="/", name="Contexto", order=0,
                   title="Contexto · Readmisión temprana")

FUENTE_CIFRA = ("Rubin, D. J. et al. (2023). Hospital Readmission Risk and Risk Factors of "
                "People with a Primary or Secondary Discharge Diagnosis of Diabetes. "
                "Journal of Clinical Medicine, 12(4), 1274.")


def _cifras():
    train, prueba = D.entrenamiento(), D.prueba()
    y = D.objetivo()
    total = len(train) + len(prueba)
    positivos = train[y].sum() + prueba[y].sum()
    pacientes = train[D.roles()["identificador"]].nunique() + prueba[D.roles()["identificador"]].nunique()
    return {"total": total, "pacientes": pacientes, "tasa": 100 * positivos / total,
            "train": len(train), "prueba": len(prueba)}


def _problema(c):
    return html.Div([
        fila_kpi([
            {"valor": pct(c["tasa"]), "etiqueta": "Readmisión en < 30 días en estos datos",
             "detalle": f"{miles(c['total'])} encuentros tras el ETL", "color": CLASE[1]},
            {"valor": "16,0 a 20,4 %", "etiqueta": "Readmisión a 30 días en diabéticos (EE. UU.)",
             "detalle": "Rubin et al. (2023)"},
            {"valor": "20.000 a 25.000 M USD", "etiqueta": "Costo anual de esas readmisiones",
             "detalle": "Rubin et al. (2023)"},
            {"valor": miles(c["pacientes"]), "etiqueta": "Pacientes distintos",
             "detalle": "Varios encuentros por paciente"},
        ]),
        dbc.Row([
            dbc.Col([
                subtitulo("Por qué importa"),
                parrafo("La diabetes es una de las enfermedades crónicas con mayor carga para los "
                        "sistemas de salud, y los pacientes diabéticos hospitalizados reingresan con "
                        "frecuencia. Una readmisión antes de 30 días se usa como indicador de la calidad "
                        "del alta: muchas son evitables con seguimiento, ajuste de la medicación o "
                        "educación al paciente, y cada una tiene un costo sanitario y humano."),
                parrafo("El servicio no puede dar seguimiento reforzado a todos los pacientes que da de "
                        "alta. La pregunta práctica es a quién priorizar: un modelo útil no tiene que "
                        "acertar con cada paciente, sino ordenar bien el riesgo para que el seguimiento "
                        "llegue primero a quienes más probablemente volverán."),
                subtitulo("Pregunta de investigación"),
                aviso("¿Con qué precisión se puede anticipar, en el momento del alta, que un paciente "
                      "diabético será readmitido antes de 30 días, a partir de los datos demográficos, "
                      "administrativos y clínicos del propio ingreso, y qué factores explican ese riesgo?",
                      tipo="info"),
            ], lg=7),
            dbc.Col([
                subtitulo("Objetivos"),
                tarjeta([
                    html.P("General", className="etiqueta-mini"),
                    parrafo("Construir y validar con rigor estadístico un modelo que priorice el "
                            "seguimiento de pacientes diabéticos según su riesgo de readmisión temprana."),
                    html.P("Específicos", className="etiqueta-mini"),
                    lista([
                        "Depurar los datos y describir los factores asociados al desenlace (ETL y EDA).",
                        "Comparar 7 modelos, 4 estrategias de balanceo y 4 optimizadores con validación "
                        "cruzada anidada y agrupada por paciente.",
                        "Proponer un modelo original y contrastarlo con los benchmarks con al menos tres "
                        "pruebas de hipótesis.",
                        "Explicar cada predicción y poner los modelos al alcance de un usuario no experto.",
                    ]),
                ], tono="panel"),
            ], lg=5),
        ], className="g-4 mt-1"),
        html.P(["Fuente de las cifras de contexto: ", FUENTE_CIFRA], className="nota-fuente"),
    ])


def _datos(c):
    r = D.roles()
    filas = []
    for grupo, variables in [("Numérica", r["numericas"]), ("Ordinal", r["ordinales"]),
                             ("Binaria", r["binarias"]), ("Categórica", r["categoricas"])]:
        for v in variables:
            filas.append({"Variable": C.nombre(v), "Columna": v, "Tipo": grupo,
                          "Descripción": C.descripcion(v)})
    filas.append({"Variable": C.nombre(r["objetivo"]), "Columna": r["objetivo"],
                  "Tipo": "Objetivo (binaria)", "Descripción": C.descripcion(r["objetivo"])})
    return html.Div([
        dbc.Row([
            dbc.Col(tarjeta(tabla(["Campo", "Valor"], [
                ["Fuente", html.A("UCI Machine Learning Repository", href=C.URL_DATOS, target="_blank")],
                ["Cita", "Clore, Cios, DeShazo y Strack (2014). Diabetes 130-US Hospitals for Years 1999-2008. UCI Machine Learning Repository."],
                ["DOI", html.A(C.DOI_DATOS, href=C.DOI_DATOS, target="_blank")],
                ["Periodo", "1999 a 2008, 130 hospitales de EE. UU."],
                ["Licencia", "Creative Commons Atribución 4.0 (CC BY 4.0)"],
                ["Registros", f"101.766 encuentros crudos; {miles(c['total'])} tras el ETL"],
                ["Variables", f"50 columnas crudas; {len(D.predictores())} predictoras tras el ETL"],
                ["Objetivo", "Readmisión en menos de 30 días (1) frente al resto (0)"],
                ["Tarea", "Clasificación binaria con clases desbalanceadas"],
                ["Partición", f"{miles(c['train'])} de entrenamiento y {miles(c['prueba'])} de prueba, "
                              "agrupada por paciente (80/20, semilla 42)"],
            ]), titulo="Ficha técnica"), lg=5),
            dbc.Col(tarjeta([
                lista([
                    "Se excluyen las altas por fallecimiento u hospicio: esos pacientes no pueden reingresar.",
                    "Se conservan todos los encuentros, pero toda validación se agrupa por paciente: los "
                    "encuentros de un mismo paciente nunca se reparten entre entrenamiento y evaluación.",
                    "El peso y el código de asegurador se eliminan; los faltantes de raza y especialidad se conservan "
                    "como categoría explícita.",
                    "Las categorías con menos del 1 % se agrupan, salvo la raza (atributo protegido).",
                ]),
                html.A([icono("menu_book", "me-1"), "Detalle en el capítulo 2 del Jupyter Book"],
                       href=C.URL_LIBRO, target="_blank", className="enlace-libro"),
            ], titulo="Decisiones del ETL", tono="panel"), lg=7),
        ], className="g-4"),
        subtitulo("Diccionario de variables"),
        dash_table.DataTable(
            data=filas, columns=[{"name": k, "id": k} for k in ["Variable", "Columna", "Tipo", "Descripción"]],
            filter_action="native", sort_action="native", page_size=12,
            style_as_list_view=True, style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "whiteSpace": "normal", "height": "auto",
                        "fontFamily": "inherit", "fontSize": "0.9rem", "padding": "8px"},
            style_header={"fontWeight": 600}, css=[{"selector": ".dash-filter input",
                                                   "rule": "font-family: inherit"}],
            filter_options={"placeholder_text": "Filtrar..."},
            id="diccionario",
        ),
    ])


PASOS = [
    ("database", "ETL", "Filtros, unidad de análisis y partición agrupada por paciente"),
    ("query_stats", "EDA", "Univariado, bivariado, redundancia y techo de desempeño"),
    ("tune", "Preprocesamiento", "Imputación, escalado robusto y one-hot dentro de cada fold"),
    ("grid_view", "112 combinaciones", "7 modelos × 4 balanceos × 4 optimizadores"),
    ("repeat", "Validación anidada", "5 folds externos × 3 internos, agrupados por paciente"),
    ("functions", "Comparación estadística", "Friedman, Nemenyi, DeLong con Holm, Cliff y Cohen"),
    ("science", "Modelo original", "Propuesta propia contrastada con los benchmarks"),
    ("verified", "Evaluación ampliada", "Datos adicionales, robustez, calibración e incertidumbre"),
    ("dashboard", "Dashboard", "Contexto, EDA y modelos en vivo"),
]


def _metodologia():
    pasos = []
    for i, (ic, titulo, texto) in enumerate(PASOS):
        pasos.append(html.Div([
            html.Div(icono(ic), className="paso-icono"),
            html.Div([html.Div(f"{i + 1:02d} · {titulo}", className="paso-titulo"),
                      html.Div(texto, className="paso-texto")]),
        ], className="paso" + (" paso-pendiente" if titulo in ("Modelo original", "Evaluación ampliada") else "")))
    return html.Div([
        html.Div(pasos, className="pipeline", role="list", **{"aria-label": "Etapas del proyecto"}),
        html.P("Las etapas con borde discontinuo corresponden al Entregable 3 y están en curso.",
               className="nota-fuente"),
        dbc.Row([
            dbc.Col(tarjeta(lista([
                "La partición de prueba se reserva desde el ETL y no interviene en ninguna decisión.",
                "Todo lo que aprende parámetros (imputación, escalado, codificación, SMOTE/ADASYN) se "
                "ajusta dentro de cada fold: no hay fuga por preprocesamiento.",
                "La métrica de selección es el AUC-PR: con un 11 % de positivos, la exactitud y el "
                "AUC-ROC son engañosos.",
                "El umbral de decisión se fija por capacidad de seguimiento: marcar al 15 % de mayor riesgo.",
            ]), titulo="Reglas que dan validez a los resultados"), lg=6),
            dbc.Col(tarjeta(lista([
                "Friedman y Nemenyi comparan los 7 modelos sobre los mismos folds.",
                "DeLong compara AUC sobre el conjunto de prueba, con corrección de Holm.",
                "Delta de Cliff y d de Cohen miden el tamaño del efecto.",
                "Intervalos BCa remuestreando pacientes completos.",
            ]), titulo="Batería estadística", tono="panel"), lg=6),
        ], className="g-4 mt-2"),
    ])


def _alcance():
    return dbc.Row([
        dbc.Col([
            subtitulo("Limitaciones"),
            lista([
                "Los datos son de 1999 a 2008 y de hospitales de EE. UU.: la práctica clínica y la "
                "población han cambiado, y el modelo no se ha validado en otro contexto.",
                "El conjunto no tiene fechas ni identificador de hospital: no se puede evaluar el "
                "desempeño en un periodo posterior ni en un centro nuevo.",
                "El techo de desempeño estimado en el EDA es moderado (AUC cercano a 0,67): las "
                "variables disponibles explican una parte limitada del riesgo.",
                "Las asociaciones no son causales: una variable importante para el modelo no es "
                "necesariamente una palanca de intervención.",
            ]),
            subtitulo("Ética y sesgo"),
            lista([
                "La raza es un atributo protegido. No se imputó y se conservó desagregada para poder "
                "auditar la equidad; el capítulo 4 registra las tasas base por grupo.",
                "Un modelo de priorización puede reproducir desigualdades de acceso presentes en los "
                "datos históricos: su uso exige revisar periódicamente las tasas de marcado por grupo.",
                "Los datos están anonimizados y se publican bajo CC BY 4.0.",
            ]),
            aviso("Este dashboard es un ejercicio académico. No es un dispositivo médico ni debe usarse "
                  "para decidir la atención de un paciente real: sirve para priorizar seguimiento en un "
                  "escenario simulado, siempre con criterio clínico.", "Uso no recomendado", "alcance"),
        ], lg=7),
        dbc.Col(tarjeta(tabla(["", ""], [
            ["Autores", html.Div([html.Div(a) for a in C.AUTORES])],
            ["Institución", C.INSTITUCION],
            ["Versión", C.VERSION],
            ["Actualizado", C.FECHA_ACTUALIZACION],
            ["Libro", html.A("Jupyter Book", href=C.URL_LIBRO, target="_blank")],
            ["Código", html.A("Repositorio en GitHub", href=C.URL_REPO, target="_blank")],
        ]), titulo="Ficha del proyecto", tono="panel"), lg=5),
    ], className="g-4")


def layout():
    c = _cifras()
    return html.Div([
        encabezado("01 · Contexto", "Readmisión temprana: el problema y los datos",
                   f"{miles(c['total'])} encuentros de pacientes diabéticos en 130 hospitales de EE. UU. "
                   "¿Quién volverá al hospital antes de 30 días?"),
        subpestanas([
            ("ctx-problema", "El problema", _problema(c)),
            ("ctx-datos", "Los datos", _datos(c)),
            ("ctx-metodo", "Metodología", _metodologia()),
            ("ctx-alcance", "Alcance y ética", _alcance()),
        ], "subpestanas-contexto"),
    ], className="pagina")
