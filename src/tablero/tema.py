"""
Tema visual: paleta validada y plantilla de Plotly.

CRITERIO DE COLOR: el color hace un trabajo o no está.
- Solo las dos clases del desenlace llevan color con significado, el mismo
  en todo el tablero. Son los del Jupyter Book con el azul ajustado: el
  #2E86AB original quedaba bajo el croma mínimo del validador de dataviz
  (0.099 < 0.10); #2A86B0 y #C1443C pasan las cinco comprobaciones sobre
  fondo claro. En modo oscuro se usan #3F92C4 y #D45C49, validados sobre
  su propio fondo (assets/style.css los sustituye en las figuras).
- Una serie sola usa COLOR_MARCA, nunca el color de una clase.
- Magnitud: rampa de un tono. Polaridad: divergente con gris en el centro.
- Variables categóricas sin significado de clase (explorador bivariado): la
  paleta categórica de referencia de dataviz, en orden fijo y siempre con
  leyenda (tres tonos quedan bajo 3:1 de contraste).
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Superficies y tinta (neutros fríos)
FONDO = "#FFFFFF"
FONDO_SUAVE = "#F2F3FA"
TINTA = "#121A3A"
TINTA_SUAVE = "#3B4266"
TINTA_TENUE = "#5D6487"
FILETE = "#E6E8F2"
REJILLA = "#ECEEF6"
EJE = "#CDD1E6"

# Clases del desenlace (0 = no readmitido, 1 = readmitido)
CLASE = {0: "#2A86B0", 1: "#C1443C"}
COLOR_MARCA = "#1B2A4E"
COLOR_TENUE = "#AEB5CC"

CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
              "#008300", "#6250d6", "#e34948"]

RAMPA = ["#F2F5FA", "#D8E3F3", "#B0C8E7", "#84A9D6", "#5C8AC4", "#3A6CAB", "#1F4E85"]
DIVERGENTE = [[0.0, "#1F4E85"], [0.25, "#84A9D6"], [0.5, "#EEEDEA"],
              [0.75, "#E59A86"], [1.0, "#A8322B"]]

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
