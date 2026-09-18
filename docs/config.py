"""
Configuración común de los notebooks del proyecto.

Contiene únicamente importaciones, constantes de reproducibilidad, estilo
gráfico y utilidades de presentación. Ninguna transformación de los datos
ocurre en este módulo: todas están en los notebooks, donde quedan
documentadas y son auditables.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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


def tabla(datos, filas=15):
    """Muestra un DataFrame como tabla interactiva si itables está instalado.

    Parameters
    ----------
    datos : pandas.DataFrame
        Tabla a mostrar.
    filas : int, default 15
        Filas visibles por página.
    """
    try:
        from itables import show
        show(datos, pageLength=filas, scrollX=True, classes="display compact")
    except ImportError:
        from IPython.display import display
        display(datos.head(filas))


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
