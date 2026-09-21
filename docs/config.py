"""
Configuración común de los notebooks del proyecto.

Contiene únicamente importaciones, constantes de reproducibilidad, estilo
gráfico y utilidades de presentación. Ninguna transformación de los datos
ocurre en este módulo: todas están en los notebooks, donde quedan
documentadas y son auditables.
"""

import json
import time
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from IPython.display import HTML

warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Reproducibilidad ------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Rutas ----------------------------------------------------------------
RAIZ = Path.cwd().parent
DATOS_CRUDOS = RAIZ / "data" / "raw"
DATOS_PROCESADOS = RAIZ / "data" / "processed"
RESULTADOS = RAIZ / "results"
for carpeta in (DATOS_PROCESADOS, RESULTADOS):
    carpeta.mkdir(parents=True, exist_ok=True)

# --- Estilo gráfico -------------------------------------------------------
PALETA = {0: "#2E86AB", 1: "#C1443C"}
COLORES = [PALETA[0], PALETA[1]]
GRIS = "#7A7A7A"

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.frameon": False,
})

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 250)


def tabla(datos, filas=15, titulo=None, indice=True):
    """
    Muestra un DataFrame con formato visual uniforme.
    """

    from IPython.display import display, HTML, Markdown

    if titulo is not None:
        display(Markdown(f"### {titulo}"))

    datos_mostrar = datos.copy()

    datos_mostrar.index.name = None

    if not indice:
        datos_mostrar = datos_mostrar.reset_index(drop=True)

    try:
        from itables import show

        show(
            datos_mostrar,
            pageLength=filas,
            scrollX=True,
            classes="display compact",
            showIndex=indice,
            columnDefs=[
                {
                    "targets": "_all",
                    "className": "dt-center"
                }
            ]
        )

        display(HTML(f"""
        <style>
            /* Contenedor principal de la tabla */
            .dataTables_wrapper {{
                width: 100% !important;
                max-width: 100% !important;
                margin: 10px 0 20px 0 !important;
            }}

            /* Tabla */
            table.dataTable {{
                width: 100% !important;
                border-collapse: collapse !important;
                font-size: 14px !important;
            }}

            /* Encabezados */
            table.dataTable thead th {{
                background-color: {PALETA[0]} !important;
                color: white !important;
                font-weight: bold !important;
                text-align: center !important;
                padding: 10px 12px !important;
                white-space: nowrap !important;
            }}

            /* Celdas */
            table.dataTable tbody td {{
                text-align: center !important;
                vertical-align: middle !important;
                padding: 9px 12px !important;
                white-space: nowrap !important;
            }}

            /* Filas alternas */
            table.dataTable tbody tr:nth-child(even) {{
                background-color: #F4F6F7 !important;
            }}

            table.dataTable tbody tr:hover {{
                background-color: #EAF2F8 !important;
            }}

            /* Bordes suaves */
            table.dataTable th,
            table.dataTable td {{
                border-bottom: 1px solid #DADADA !important;
            }}

            /* Evita que el buscador y controles se compriman */
            .dataTables_wrapper .dataTables_filter,
            .dataTables_wrapper .dataTables_length {{
                margin-bottom: 8px !important;
            }}
        </style>
        """))

    except ImportError:
        tabla_estilizada = (
            datos_mostrar.head(filas)
            .style
            .set_properties(**{
                "text-align": "center",
                "white-space": "nowrap",
                "font-size": "11pt",
                "padding": "9px 12px",
                "vertical-align": "middle"
            })
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", PALETA[0]),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("white-space", "nowrap")
                    ]
                },
                {
                    "selector": "table",
                    "props": [
                        ("width", "100%"),
                        ("table-layout", "auto"),
                        ("border-collapse", "collapse")
                    ]
                },
                {
                    "selector": "tbody tr:nth-child(even)",
                    "props": [
                        ("background-color", "#F4F6F7")
                    ]
                }
            ])
        )

        display(HTML(f"""
        <div style="
            width: 100%;
            max-height: 400px;
            overflow-x: auto;
            overflow-y: auto;
            border: 1px solid {GRIS};
            border-radius: 8px;
            margin: 10px 0 20px 0;
        ">
            {tabla_estilizada.to_html()}
        </div>
        """))


def tabla_estilizada(
    datos,
    filas=5,
    titulo=None,
    indice=True,
    posicion="inicio"
):
    """
    Muestra las primeras o últimas filas de un DataFrame con estilo.

    posicion : str, default="inicio"
        "inicio" para las primeras filas.
        "final" para las últimas filas.
    """

    from IPython.display import display, HTML

    if posicion == "final":
        datos_mostrar = datos.tail(filas).copy()
    else:
        datos_mostrar = datos.head(filas).copy()

    if not indice:
        datos_mostrar = datos_mostrar.reset_index(drop=True)

    tabla_formateada = (
        datos_mostrar.style
        .set_caption(titulo if titulo else "")
        .set_properties(**{
            "text-align": "center",
            "white-space": "nowrap",
            "font-size": "11pt",
            "padding": "8px"
        })
        .set_table_styles([
            {
                "selector": "caption",
                "props": [
                    ("font-size", "16px"),
                    ("font-weight", "bold"),
                    ("color", PALETA[0]),
                    ("padding", "10px")
                ]
            },
            {
                "selector": "th",
                "props": [
                    ("background-color", PALETA[0]),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "8px")
                ]
            },
            {
                "selector": "table",
                "props": [
                    ("width", "100%"),
                    ("table-layout", "auto"),
                    ("border-collapse", "collapse")
                ]
            }
        ])
    )

    display(HTML(f"""
    <div style="
        max-width: 100%;
        max-height: 400px;
        overflow: auto;
        border: 1px solid {GRIS};
        border-radius: 8px;
        margin: 10px 0;
    ">
        {tabla_formateada.to_html()}
    </div>
    """))



def columnas_texto(datos):
    """Devuelve las columnas de tipo texto de un DataFrame.

    Evita ``select_dtypes(include="object")``, que en pandas 3 selecciona
    también las columnas de tipo ``str`` mostrando un aviso de deprecación y
    dejará de hacerlo en pandas 4. Esta versión funciona igual en pandas 2 y 3.

    Parameters
    ----------
    datos : pandas.DataFrame

    Returns
    -------
    list of str
        Nombres de las columnas de texto, en el orden del DataFrame.
    """
    return [c for c in datos.columns
            if datos[c].dtype == object or pd.api.types.is_string_dtype(datos[c])]


def guardar_resultado(datos, nombre):
    """Persiste una tabla de resultados en results/ para trazabilidad."""
    destino = RESULTADOS / f"{nombre}.csv"
    datos.to_csv(destino)
    return destino


def leer_tabla(nombre):
    """Lee un conjunto procesado, con parquet como formato preferido."""
    parquet = DATOS_PROCESADOS / f"{nombre}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    # keep_default_na=False evita que pandas convierta en nulo cadenas como
    # "None" o "NA" que en este proyecto son categorías legítimas. El
    # conjunto depurado no contiene faltantes, así que no se pierde nada.
    return pd.read_csv(DATOS_PROCESADOS / f"{nombre}.csv",
                       keep_default_na=False, na_values=[])


def escribir_tabla(datos, nombre):
    """Escribe un conjunto procesado en parquet, con respaldo en CSV."""
    try:
        destino = DATOS_PROCESADOS / f"{nombre}.parquet"
        datos.to_parquet(destino, index=False)
    except (ImportError, ValueError):
        destino = DATOS_PROCESADOS / f"{nombre}.csv"
        datos.to_csv(destino, index=False)
    return destino


def roles_variables(nombre="roles_variables.json"):
    """Lee el diccionario de roles de variables definido en el notebook 02."""
    with open(DATOS_PROCESADOS / nombre, encoding="utf-8") as f:
        return json.load(f)
