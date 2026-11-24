"""
Figuras de la página /modelo-original (el relato del modelo original).

Cada función recibe tablas ya calculadas (``explicador``, ``evidencia``) y
devuelve una figura; las páginas no estilan gráficos. Solo usan colores que ya
están en ``tema.py`` (así las reglas de modo oscuro de ``style.css`` los
cubren). Cada modelo lleva el color de su papel (``tema.color_modelo``, igual en todo
el tablero): DES-CPL dorado, XGBoost azul, ensambles estáticos en un tono común y
el resto gris; los pacientes readmitidos y no readmitidos conservan los colores de
clase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config as C
from .evidencia import EQUIVALENTE, MEJOR, ORIGINAL, PEOR, SIN_CONCLUSION
from .formato import num
from .tema import (CLASE, COLOR_MARCA, COLOR_SEGUNDO, COLOR_TENUE, EJE, ENSAMBLES_ESTATICOS, FONDO, REJILLA,
                   TINTA, TINTA_TENUE, color_modelo, tematizar)

ESTATICOS = set(ENSAMBLES_ESTATICOS)


def color_metodo(clave_o_nombre: str) -> str:
    """Color del modelo según su papel; lo decide ``tema.color_modelo`` para todo el tablero.

    Acepta la clave del tablero («des_cpl») o el nombre que traen los CSV.
    """
    return color_modelo(clave_o_nombre)


def _rejilla(fig, eje="x"):
    getattr(fig, f"update_{eje}axes")(showgrid=True, gridcolor=REJILLA)
    return fig


# =========================================================================== #
# Paso 2: cómo decide
# =========================================================================== #
def fig_proyeccion(caso: dict, proyeccion: dict):
    """Región de competencia en la proyección 2D del espacio de perfiles.

    Fondo: una muestra de los pacientes del conjunto de selección. Resaltados:
    los ``k`` vecinos positivos (readmitidos), los ``k`` negativos y el paciente.
    """
    fondo = np.asarray(proyeccion["fondo"])
    vecinos = np.asarray(caso["vecinos_xy"])
    y = np.asarray(caso["vecinos_y"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fondo[:, 0], y=fondo[:, 1], mode="markers", name="Pacientes de selección",
                               marker=dict(size=4, color=COLOR_TENUE, opacity=0.45), hoverinfo="skip"))
    for clase, nombre in ((0, "Vecinos no readmitidos"), (1, "Vecinos readmitidos")):
        sel = vecinos[y == clase]
        fig.add_trace(go.Scatter(x=sel[:, 0], y=sel[:, 1], mode="markers", name=nombre,
                                 marker=dict(size=9, color=CLASE[clase], line=dict(color=FONDO, width=1)),
                                 hovertemplate=f"{nombre}<extra></extra>"))
    px, py = caso["paciente_xy"]
    fig.add_trace(go.Scatter(x=[px], y=[py], mode="markers", name="Este paciente",
                             marker=dict(size=15, color=TINTA, symbol="diamond", line=dict(color=FONDO, width=2)),
                             hovertemplate="Este paciente<extra></extra>"))
    v1, v2 = proyeccion["varianza"]
    fig.update_xaxes(title_text=f"Componente 1 ({num(100 * v1, 0)} % de la varianza de los logits)")
    fig.update_yaxes(title_text=f"Componente 2 ({num(100 * v2, 0)} %)", showgrid=True, gridcolor=REJILLA)
    return tematizar(fig, alto=380, margin=dict(l=56, r=16, t=30, b=48),
                     legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0))


def fig_competencias(caso: dict, mezcla: list[float]):
    """Competencia local y global de cada modelo, medidas sobre el mejor.

    La competencia es un Brier ponderado (menor es mejor) y las diferencias
    entre modelos son de milésimas, así que cada barra muestra cuánto peor es
    ese modelo que el mejor *de esa misma medida*. El rombo es la competencia
    que usa DES-CPL con el λ elegido.
    """
    local = np.asarray(caso["competencia_local"])
    glob = np.asarray(caso["competencia_global"])
    usada = np.asarray(mezcla)
    escala = 1000.0
    nombres = caso["modelos"]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=nombres, x=(local - local.min()) * escala, orientation="h", name="Local (sus vecinos)",
                         marker_color=COLOR_SEGUNDO, hovertemplate="%{y}: +%{x:.3f}<extra>Local</extra>"))
    fig.add_trace(go.Bar(y=nombres, x=(glob - glob.min()) * escala, orientation="h", name="Global (todos los pacientes)",
                         marker_color=COLOR_MARCA, hovertemplate="%{y}: +%{x:.3f}<extra>Global</extra>"))
    fig.add_trace(go.Scatter(y=nombres, x=(usada - usada.min()) * escala, mode="markers", name="Usada (mezcla con λ)",
                             marker=dict(size=11, color=TINTA, symbol="diamond", line=dict(color=FONDO, width=1.5)),
                             hovertemplate="%{y}: +%{x:.3f}<extra>Usada</extra>"))
    fig.update_xaxes(title_text="Brier sobre el mejor modelo<br>(× 10⁻³; menos es mejor)", showgrid=True,
                     gridcolor=REJILLA, rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    return tematizar(fig, alto=300, barmode="group", margin=dict(l=130, r=16, t=30, b=72),
                     legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0))


# =========================================================================== #
# Paso 3: el resultado
# =========================================================================== #
def fig_por_fold(largo: pd.DataFrame):
    """AUC-PR por fold de cada método (puntos) con la media ± desviación."""
    resumen = largo.groupby("clave")["auc_pr"].agg(["mean", "std"]).sort_values("mean")
    nombres = {c: C.nombre_modelo(c) for c in resumen.index}
    fig = go.Figure()
    for clave in resumen.index:
        datos = largo[largo["clave"] == clave]
        fig.add_trace(go.Scatter(x=datos["auc_pr"], y=[nombres[clave]] * len(datos), mode="markers",
                                 marker=dict(size=7, color=color_metodo(clave), opacity=0.45, symbol="circle-open",
                                             line=dict(width=1.5)),
                                 showlegend=False, customdata=datos["fold"] + 1,
                                 hovertemplate="%{y}, fold %{customdata}: %{x:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=resumen["mean"], y=[nombres[c] for c in resumen.index], mode="markers", name="Media ± desviación",
        error_x=dict(type="data", array=resumen["std"], color=EJE, thickness=2, width=4),
        marker=dict(size=[14 if c == ORIGINAL else 10 for c in resumen.index],
                    color=[color_metodo(c) for c in resumen.index], line=dict(color=FONDO, width=2)),
        customdata=resumen["std"], showlegend=False,
        hovertemplate="%{y}: %{x:.4f} ± %{customdata:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="AUC-PR por fold externo (círculos) y media ± desviación", showgrid=True,
                     gridcolor=REJILLA)
    return tematizar(fig, alto=470, margin=dict(l=170, r=24, t=16, b=52), showlegend=False)


#: Texto y color de cada categoría del intervalo frente a la franja de equivalencia.
CATEGORIAS = {MEJOR: ("DES-CPL mejor (el intervalo excluye el 0)", COLOR_MARCA),
              PEOR: ("DES-CPL peor (el intervalo excluye el 0)", CLASE[1]),
              EQUIVALENTE: ("Equivalente (el intervalo cae dentro de la franja)", COLOR_SEGUNDO),
              SIN_CONCLUSION: ("Sin conclusión (el intervalo cruza el 0)", COLOR_TENUE)}


def fig_diferencias(tabla: pd.DataFrame, equivalencia: float = C.EQUIVALENCIA_AUC_PR):
    """DES-CPL − cada benchmark, con el intervalo del 95 % y la franja de equivalencia."""
    tabla = tabla.sort_values("diferencia")
    fig = go.Figure()
    fig.add_vrect(x0=-equivalencia, x1=equivalencia, fillcolor=COLOR_SEGUNDO, opacity=0.18, line_width=0,
                  annotation_text=f"franja ±{num(equivalencia, 3)}", annotation_position="top left",
                  annotation_font=dict(size=11, color=TINTA_TENUE))
    fig.add_vline(x=0, line=dict(color=TINTA_TENUE, width=1))
    for categoria, (texto, color) in CATEGORIAS.items():
        t = tabla[tabla["categoria"] == categoria]
        if t.empty:
            continue
        fig.add_trace(go.Scatter(
            x=t["diferencia"], y=t["benchmark"], mode="markers", name=texto,
            error_x=dict(type="data", symmetric=False, array=t["ic_alto"] - t["diferencia"],
                         arrayminus=t["diferencia"] - t["ic_bajo"], color=color, thickness=2, width=5),
            marker=dict(size=11, color=color, line=dict(color=FONDO, width=2)),
            customdata=np.column_stack([t["ic_bajo"], t["ic_alto"], t["p"], t["p_holm"]]),
            hovertemplate=("%{y}: %{x:+.4f}<br>IC 95 %: %{customdata[0]:+.4f} a %{customdata[1]:+.4f}"
                           "<br>p = %{customdata[2]:.3f}, Holm %{customdata[3]:.3f}<extra></extra>")))
    fig.update_xaxes(title_text="Diferencia de AUC-PR: DES-CPL − benchmark (a la derecha, DES-CPL mejor)",
                     showgrid=True, gridcolor=REJILLA, zeroline=False)
    return tematizar(fig, alto=400, margin=dict(l=150, r=24, t=34, b=60),
                     legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0))


# =========================================================================== #
# Paso 4: por qué empata
# =========================================================================== #
def fig_lambda_fold(lambdas: pd.Series):
    """El λ elegido en cada fold, como termómetro de la heterogeneidad local.

    λ = 0 es la selección dinámica pura (confiar en la región del paciente);
    λ = 1, pesos globales (ignorar la región). Cuanto más a la derecha, menos
    aporta la parte local.
    """
    folds = [f"Fold {int(f) + 1}" for f in lambdas.index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lambdas.to_numpy(), y=folds, mode="markers+text", text=[num(v, 2) for v in lambdas],
                             textposition="top center", textfont=dict(color=TINTA, size=12),
                             marker=dict(size=15, color=color_modelo(ORIGINAL), line=dict(color=FONDO, width=2)),
                             hovertemplate="%{y}: λ = %{x:.2f}<extra></extra>", showlegend=False))
    for fila in folds:
        fig.add_shape(type="line", x0=0, x1=1, y0=fila, y1=fila, line=dict(color=EJE, width=3), layer="below")
    fig.update_xaxes(range=[-0.12, 1.12], tickvals=[0, 0.25, 0.5, 0.75, 1], showgrid=False,
                     title_text="λ elegido: 0 = confiar en la región del paciente · 1 = pesos globales")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return tematizar(fig, alto=280, margin=dict(l=70, r=24, t=16, b=56))


def fig_ablacion(tabla: pd.DataFrame):
    """Diferencia de AUC-PR de cada variante frente a DES-CPL completo (media ± desviación)."""
    tabla = tabla.sort_values("diferencia", ascending=False)
    nombres = [v.replace("Ablación: ", "").replace("DES-CPL inicial", "Versión inicial") for v in tabla["variante"]]
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color=TINTA_TENUE, width=1))
    fig.add_trace(go.Scatter(
        x=tabla["diferencia"], y=nombres, mode="markers",
        error_x=dict(type="data", array=tabla["sd"], color=EJE, thickness=2, width=4),
        marker=dict(size=11, color=color_modelo(ORIGINAL), line=dict(color=FONDO, width=2)), customdata=tabla["sd"],
        hovertemplate="%{y}: %{x:+.4f} ± %{customdata:.4f}<extra></extra>", showlegend=False))
    fig.update_xaxes(title_text="Variante − DES-CPL completo (AUC-PR, media ± desv.)",
                     showgrid=True, gridcolor=REJILLA, zeroline=False)
    fig.update_yaxes(autorange="reversed")
    return tematizar(fig, alto=380, margin=dict(l=210, r=24, t=16, b=56))


# =========================================================================== #
# Paso 5: generalización y estabilidad
# =========================================================================== #
def fig_rangos_conjuntos(rangos: pd.DataFrame):
    """Rango medio en los 4 conjuntos (barra) y en los 3 adicionales (rombo)."""
    rangos = rangos.sort_values("rango_4", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=rangos["metodo"], x=rangos["rango_4"], orientation="h", name="Los 4 conjuntos",
                         marker_color=[color_metodo(c) for c in rangos["clave"]],
                         hovertemplate="%{y}: rango medio %{x:.2f}<extra>4 conjuntos</extra>"))
    fig.add_trace(go.Scatter(y=rangos["metodo"], x=rangos["rango_3"], mode="markers", name="Solo los 3 adicionales",
                             marker=dict(size=9, color=TINTA, symbol="diamond", line=dict(color=FONDO, width=1.5)),
                             hovertemplate="%{y}: rango medio %{x:.2f}<extra>3 adicionales</extra>"))
    fig.update_xaxes(title_text="Rango medio de AUC-PR (1 = mejor)", showgrid=True, gridcolor=REJILLA)
    fig.update_yaxes(autorange="reversed")
    return tematizar(fig, alto=470, margin=dict(l=170, r=24, t=30, b=52),
                     legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0))


def fig_robustez(largo: pd.DataFrame, metodos: list[str]):
    """Pérdida de AUC-PR en cada escenario de robustez, un punto por método."""
    from .evidencia import ESCENARIOS

    escenarios = [e for e in ESCENARIOS if e in set(largo["escenario"])]
    etiquetas = [ESCENARIOS[e] for e in escenarios]
    fig = go.Figure()
    for metodo in metodos:
        datos = largo[largo["método"] == metodo].set_index("escenario").loc[escenarios]
        clave = ORIGINAL if metodo == "DES-CPL" else metodo
        nombre = C.nombre_modelo(clave)
        fig.add_trace(go.Scatter(
            x=datos["perdida"], y=etiquetas, mode="markers", name=nombre,
            marker=dict(size=13 if clave == ORIGINAL else 10, color=color_metodo(nombre),
                        symbol="diamond" if clave == "xgboost" else "circle", line=dict(color=FONDO, width=1.5)),
            hovertemplate=f"{nombre}: pierde %{{x:.4f}}<extra></extra>"))
    fig.update_xaxes(title_text="AUC-PR que pierde frente al problema original (menos es mejor)", showgrid=True,
                     gridcolor=REJILLA, rangemode="tozero")
    fig.update_yaxes(autorange="reversed")
    return tematizar(fig, alto=340, margin=dict(l=190, r=24, t=30, b=56),
                     legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0))


def fig_semillas(resumen: pd.DataFrame):
    """AUC-PR en 10 semillas: rango (mín–máx) y media ± desviación."""
    resumen = resumen.sort_values("mean")
    fig = go.Figure()
    for _, f in resumen.iterrows():
        fig.add_trace(go.Scatter(x=[f["min"], f["max"]], y=[f["metodo"]] * 2, mode="lines", showlegend=False,
                                 line=dict(color=EJE, width=6), hoverinfo="skip"))
    clave = [("des_cpl" if m == C.nombre_modelo(ORIGINAL) else m) for m in resumen["metodo"]]
    fig.add_trace(go.Scatter(
        x=resumen["mean"], y=resumen["metodo"], mode="markers", showlegend=False,
        error_x=dict(type="data", array=resumen["std"], color=TINTA_TENUE, thickness=2, width=4),
        marker=dict(size=12, color=[color_metodo(m) for m in resumen["metodo"]],
                    line=dict(color=FONDO, width=2)),
        customdata=np.column_stack([resumen["std"], resumen["min"], resumen["max"]]),
        hovertemplate="%{y}: %{x:.4f} ± %{customdata[0]:.4f}<br>mín. %{customdata[1]:.4f}, máx. %{customdata[2]:.4f}"
                      "<extra></extra>"))
    fig.update_xaxes(title_text="AUC-PR en 10 semillas: rango (barra) y media ± desviación (punto)",
                     showgrid=True, gridcolor=REJILLA)
    return tematizar(fig, alto=280, margin=dict(l=170, r=24, t=16, b=56))
