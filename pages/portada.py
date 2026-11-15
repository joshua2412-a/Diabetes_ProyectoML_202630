"""
Portada del dashboard: la sala de espera animada de assets/portada.html.

Ocupa toda la ventana (tapa la barra y el pie) y su botón «Entrar al
dashboard» abre /contexto. La portada es un HTML estático en un iframe del
mismo origen: lee y cambia el mismo tema claro u oscuro que el tablero
(localStorage «tema»), así que pasar de una a otro no destella.
"""

from __future__ import annotations

import dash
from dash import html

dash.register_page(__name__, path="/", name="Portada", order=-1,
                   title="Readmisión temprana en diabéticos")


def layout():
    return html.Iframe(src="/assets/portada.html?to=/contexto", className="portada-marco",
                       title="Portada: readmisión temprana en pacientes diabéticos")
