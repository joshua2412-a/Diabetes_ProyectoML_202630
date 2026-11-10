"""
Dashboard predictivo de readmisión temprana en pacientes diabéticos.

Aplicación multipágina (Dash pages) con las tres secciones que exige el
Entregable 3: Contexto, EDA y Modelos. Este archivo solo crea la aplicación,
la barra de navegación y el pie; el contenido vive en pages/*.py y la lógica
en src/tablero/*.py. El dashboard no entrena: lee artifacts/.

Local:    python app.py            -> http://127.0.0.1:8050
Render:   gunicorn app:server      (ver render.yaml)
"""

from __future__ import annotations

import threading

import dash
import dash_bootstrap_components as dbc
from dash import html

from src.tablero import config as C
from src.tablero import datos as D
from src.tablero.componentes import icono

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap",
        # display=block: que no se vea el nombre del icono en texto mientras carga
        "https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:"
        "opsz,wght,FILL,GRAD@20..48,300..600,0..1,0&display=block",
    ],
    suppress_callback_exceptions=True,   # las subpestañas montan sus componentes después
    title="Readmisión temprana en diabéticos",
    update_title="Calculando...",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # objeto WSGI que sirve gunicorn en Render

# El tema (claro u oscuro) se fija en el <head>, antes del primer pintado: si
# lo pusiera un .js de assets/, que Dash carga al final, la página destellaría.
app.index_string = """<!DOCTYPE html>
<html lang="es">
    <head>
        {%metas%}<title>{%title%}</title>{%favicon%}
        <script>
            (function () {
                var g = null; try { g = localStorage.getItem("tema"); } catch (e) {}
                var oscuro = g ? g === "oscuro"
                    : !!(window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches);
                document.documentElement.setAttribute("data-tema", oscuro ? "oscuro" : "claro");
            })();
        </script>
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>"""


def _precalentar() -> None:
    """Carga datos y metadatos al arrancar, para que el primer clic no espere."""
    try:
        D.entrenamiento()
        D.opciones_filtro()
        for modelo in D.modelos_disponibles():
            D.metadata(modelo)
    except Exception as error:  # noqa: BLE001 - la página mostrará el aviso
        print(f"[arranque] No se pudieron precargar los datos: {error}")


threading.Thread(target=_precalentar, daemon=True).start()


def barra() -> html.Div:
    enlaces = [dbc.NavLink([icono(i, "me-2"), n], href=r, active="exact", className="barra-enlace")
               for n, r, i in [("Contexto", "/", "public"), ("EDA", "/eda", "bar_chart"),
                               ("Modelos", "/modelos", "account_tree")]]
    return html.Div(dbc.Container([
        html.A([html.Span("Readmisión temprana", className="barra-marca-titulo"),
                html.Span("Diabetes 130-US Hospitals", className="barra-marca-sub")],
               href="/", className="barra-marca"),
        dbc.Nav(enlaces, className="barra-nav", pills=False),
        html.Div([
            html.Button([icono("dark_mode", "tema-icono tema-luna"), icono("light_mode", "tema-icono tema-sol")],
                        id="boton-tema", className="boton-barra", type="button",
                        title="Cambiar entre modo claro y oscuro",
                        **{"aria-label": "Cambiar entre modo claro y oscuro"}),
            html.A([icono("menu_book", "me-1"), html.Span("Jupyter Book", className="solo-escritorio")],
                   href=C.URL_LIBRO, target="_blank", rel="noopener noreferrer", className="boton-barra"),
        ], className="barra-acciones"),
    ], fluid=True, className="barra-contenido"), className="barra")


app.layout = html.Div([
    barra(),
    html.Main(dash.page_container, className="contenido"),
    html.Footer(dbc.Container([
        html.Span(f"{' y '.join(C.AUTORES)} · {C.INSTITUCION}"),
        html.Span(f"Versión {C.VERSION} · {C.FECHA_ACTUALIZACION} · Datos: Clore et al. (2014), CC BY 4.0"),
    ], fluid=True, className="pie-contenido"), className="pie"),
], className="app-root")


if __name__ == "__main__":
    # Solo en local. En Render arranca gunicorn con `app:server`.
    print("\nDashboard disponible en http://127.0.0.1:8050\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
