"""
Tema visual: paleta validada y plantilla de Plotly.

PALETA DEL PROYECTO (cinco colores, el máximo que pide la guía):
Golden Sand #E8C766, Amber Blaze #E8AF2E, Azure Crest #3462AC,
Deep Midnight #223164 y Shadow Abyss #0E2043. Los dos azules noche son la
banda superior y la tinta; el dorado y el ámbar son acentos de interfaz.

CRITERIO DE COLOR: el color hace un trabajo o no está.
- Solo las dos clases del desenlace llevan color con significado, el mismo
  en todo el tablero: azul (no readmitido) y ámbar (readmitido). Azul frente
  a ámbar se distingue con cualquier tipo de daltonismo, cosa que el par
  azul/rojo anterior no garantizaba igual de bien.
- El ámbar puro (#E8AF2E) es demasiado claro para marcar datos sobre blanco
  (contraste 1,9:1 y fuera de la banda de luminosidad del validador de
  dataviz), así que en los gráficos se usa el mismo tono oscurecido:
  claro #3462AC / #B97C0C, oscuro #5A8FD8 / #C2850C (sobre #0E2043). Los dos
  pares pasan las cinco comprobaciones de scripts/validate_palette.js de la
  skill dataviz. assets/style.css sustituye los del modo oscuro.
- Una serie sola usa COLOR_MARCA (Deep Midnight), nunca el color de una clase.
- Magnitud: rampa de un tono (Azure Crest). Polaridad: divergente azul-ámbar
  con gris en el centro.
- Variables categóricas sin significado de clase (explorador bivariado): la
  paleta categórica de referencia de dataviz, en orden fijo y siempre con
  leyenda; es el único sitio donde aparecen más tonos.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Paleta del proyecto
ARENA = "#E8C766"        # Golden Sand
AMBAR = "#E8AF2E"        # Amber Blaze
AZUR = "#3462AC"         # Azure Crest
MEDIANOCHE = "#223164"   # Deep Midnight
ABISMO = "#0E2043"       # Shadow Abyss

# Superficies y tinta (neutros fríos, derivados de la paleta)
FONDO = "#FFFFFF"
FONDO_SUAVE = "#F2F4FA"
TINTA = ABISMO
TINTA_SUAVE = "#3B4266"
TINTA_TENUE = "#5D6487"
FILETE = "#E6E8F2"
REJILLA = "#ECEEF6"
EJE = "#CDD1E6"

# Clases del desenlace (0 = no readmitido, 1 = readmitido)
CLASE = {0: AZUR, 1: "#B97C0C"}
COLOR_MARCA = MEDIANOCHE
COLOR_TENUE = "#AEB5CC"
#: Segundo grupo en comparaciones A frente a B (no es una clase).
COLOR_SEGUNDO = "#7FA0D6"

CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
              "#008300", "#6250d6", "#e34948"]

RAMPA = ["#F1F4FA", "#D6E0F2", "#B2C5E8", "#85A4D6", "#5A85C4", AZUR, MEDIANOCHE]
DIVERGENTE = [[0.0, MEDIANOCHE], [0.25, "#85A4D6"], [0.5, "#EEEDEA"],
              [0.75, "#E3BE6E"], [1.0, "#9A6508"]]
ESCALA_RAMPA = [[i / (len(RAMPA) - 1), c] for i, c in enumerate(RAMPA)]
#: Magnitud de la tasa de readmisión: rampa de un tono en el ámbar de la
#: clase «readmitido» (de Golden Sand aclarado al ámbar oscuro).
RAMPA_AMBAR = ["#F4E0A9", "#E8C766", "#D9A23A", "#B97C0C", "#7E5405"]
ESCALA_AMBAR = [[i / (len(RAMPA_AMBAR) - 1), c] for i, c in enumerate(RAMPA_AMBAR)]


def rgba(color: str, alfa: float) -> str:
    """Hex a rgba: los rellenos translúcidos (bandas, flujos) usan el color
    de la clase sin inventar un tono nuevo."""
    color = color.lstrip("#")
    r, g, b = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alfa})"

ESTADO_BIEN = "#1F7A43"
ESTADO_AVISO = "#9A6B08"
ESTADO_CRITICO = "#B3261E"

FUENTE = '"Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

PLANTILLA = go.layout.Template(layout=go.Layout(
    colorway=[COLOR_MARCA],
    separators=",.",
    font=dict(family=FUENTE, size=13, color=TINTA_SUAVE),
    title=dict(font=dict(size=14, color=TINTA), x=0, xanchor="left"),
    paper_bgcolor=FONDO, plot_bgcolor=FONDO,
    margin=dict(l=56, r=20, t=40, b=48),
    hoverlabel=dict(bgcolor=FONDO, bordercolor=FILETE,
                    font=dict(family=FUENTE, size=12, color=TINTA)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                title_text="", bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=TINTA_SUAVE)),
    xaxis=dict(showgrid=False, zeroline=False, linecolor=EJE, ticks="outside", ticklen=4,
               tickcolor=EJE, tickfont=dict(color=TINTA_TENUE, size=12),
               title_font=dict(color=TINTA_SUAVE, size=12), automargin=True),
    yaxis=dict(gridcolor=REJILLA, zeroline=False, linecolor="rgba(0,0,0,0)", ticks="",
               tickfont=dict(color=TINTA_TENUE, size=12),
               title_font=dict(color=TINTA_SUAVE, size=12), automargin=True),
))
pio.templates["readmision"] = PLANTILLA
pio.templates.default = "readmision"


def tematizar(fig: go.Figure, alto: int | None = None, **layout) -> go.Figure:
    """Último paso de toda figura: plantilla, altura y ajustes comunes."""
    fig.update_layout(template=PLANTILLA, **layout)
    if alto is not None:
        fig.update_layout(height=alto)
    return fig


def con_guia(fig: go.Figure) -> go.Figure:
    """Línea guía vertical punteada que sigue al cursor (como en un monitor).

    Se usa en las curvas que se leen de izquierda a derecha (dosis-respuesta,
    ganancia, «qué pasaría si»): ayuda a leer el valor sin buscar la marca.
    """
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="data", spikedash="dot",
                     spikethickness=1, spikecolor=TINTA_TENUE)
    fig.update_layout(hoverdistance=40, spikedistance=-1)
    return fig
