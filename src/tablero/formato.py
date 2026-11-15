"""
Formato en español: el único sitio donde se decide cómo se escribe un número.

Coma decimal, punto de millar y espacio antes de «%». Las figuras de Plotly
no pasan por aquí: lo resuelve ``separators=",."`` en la plantilla.
"""

from __future__ import annotations

import math
import re


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


#: Número con punto decimal que no forma parte de una ruta, un DOI ni una
#: lista de versiones (1.2.3), y que no es un número de sección o de versión.
#: Puede ir seguido de un punto final o de un exponente (1.30e-05).
_DECIMAL = re.compile(r"(?<![\w/.])(?<!sección )(?<!capítulo )(?<!Versión:\*\* )(\d+)\.(\d+)(?=$|[^\w/.]|\.(?!\d)|e[-+]?\d)")


def decimales_es(texto: str) -> str:
    """Pasa a coma decimal un texto escrito con punto (Markdown del libro o
    de artifacts/, que siguen la convención del libro).

    Dentro de las fórmulas ($...$) escribe {,} para que MathJax no deje el
    hueco que pone tras una coma. No toca números de sección ni la versión.
    """
    partes = texto.split("$")
    for i, parte in enumerate(partes):
        partes[i] = _DECIMAL.sub(r"\1{,}\2" if i % 2 else r"\1,\2", parte)
    return "$".join(partes)
