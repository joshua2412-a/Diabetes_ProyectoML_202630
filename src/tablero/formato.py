"""
Formato en español: el único sitio donde se decide cómo se escribe un número.

Coma decimal, punto de millar y espacio antes de «%». Las figuras de Plotly
no pasan por aquí: lo resuelve ``separators=",."`` en la plantilla.
"""

from __future__ import annotations

import math


def _nulo(valor) -> bool:
    return valor is None or (isinstance(valor, float) and math.isnan(valor))


def num(valor, decimales: int = 1, signo: bool = False) -> str:
    if _nulo(valor):
        return "–"
    texto = f"{valor:+,.{decimales}f}" if signo else f"{valor:,.{decimales}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def miles(valor) -> str:
    if _nulo(valor):
        return "–"
    return f"{int(round(valor)):,}".replace(",", ".")


def pct(valor, decimales: int = 1, signo: bool = False) -> str:
    """Porcentaje a partir de un valor en escala 0-100: 11.39 -> «11,4 %»."""
    if _nulo(valor):
        return "–"
    return f"{num(valor, decimales, signo)} %"


def p_valor(valor, umbral: float = 0.0001) -> str:
    if _nulo(valor):
        return "–"
    return "p < 0,0001" if valor < umbral else f"p = {num(valor, 4)}"


def tnum(valor, decimales: int = 1) -> str:
    """``num`` dentro de LaTeX: 0,10 -> 0{,}10."""
    return num(valor, decimales).replace(",", "{,}")


def lista_y(elementos) -> str:
    partes = [str(e) for e in elementos if e]
    if len(partes) <= 1:
        return "".join(partes)
    return ", ".join(partes[:-1]) + " y " + partes[-1]
