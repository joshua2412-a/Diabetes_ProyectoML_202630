"""
Lectura de los resultados de Google Scholar copiados a texto (capítulo 14).

Google Scholar no tiene API y bloquea las consultas automatizadas. Por eso
la búsqueda se hace a mano en el navegador: para cada cadena se copia la
página de resultados (Ctrl+A, Ctrl+C) y se pega en
``results/literatura/google_scholar/<cadena>.txt``. Si hay varias páginas,
se pegan una debajo de otra en el mismo archivo.

Este módulo convierte esos archivos en las mismas dos tablas que produce
``experiments/revision_literatura.py`` para las bases con API: una fila por
búsqueda (con el total de resultados) y una fila por registro (año, título,
fuente y fragmento).

Cada resultado de Google Scholar ocupa varias líneas y termina en la línea
de acciones («Guardar Citar …»). La línea de autores tiene la forma
«autores - revista, año - dominio»; el título es la línea anterior.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

ACCIONES = re.compile(r"^(Guardar|Save)\s+(Citar|Cite)\b")
AUTORES = re.compile(r"^(?P<autores>.+?) - (?P<fuente>.*?)\b(?P<anio>(19|20)\d{2})\b.*? - (?P<dominio>.+?)\s*$")
AUTORES_SIN_ANIO = re.compile(r"^(?P<autores>.+?) - (?P<dominio>\S+\.\S+)\s*$")
LECTURA_RAPIDA = re.compile(r"^(Lectura rápida|Quick view)\b")
PREFIJO = re.compile(r"^(\[(HTML|PDF|LIBRO|BOOK|CITAS|CITATION|DOC|TXT)\]\s*)+")
TOTAL = re.compile(r"^(?:Aproximadamente |About )?(?P<n>[\d.,]+) resultados?\b", re.IGNORECASE)
PAGINA = re.compile(r"^(?:Página|Page) \d+ (?:de|of) (?:aproximadamente |about )?(?P<n>[\d.,]+) resultados?\b",
                    re.IGNORECASE)
SIN_RESULTADOS = re.compile(r"no ha obtenido ningún|did not match any", re.IGNORECASE)


def _total(lineas: list[str]) -> int | None:
    """Total de resultados (None si no aparece).

    La primera página da una estimación («Aproximadamente N resultados»); las
    siguientes («Página k de N resultados») la corrigen. Se usa la última.
    """
    total = None
    for linea in lineas:
        if SIN_RESULTADOS.search(linea):
            return 0
        coincidencia = TOTAL.match(linea.strip()) or PAGINA.match(linea.strip())
        if coincidencia:
            total = int(re.sub(r"[.,]", "", coincidencia["n"]))
    return total


def _registros(lineas: list[str]) -> list[dict]:
    """Un diccionario por resultado: título, autores, fuente, año y fragmento."""
    registros, bloque = [], []
    for linea in lineas:
        linea = linea.strip()
        if ACCIONES.match(linea):
            registro = _leer_bloque(bloque)
            if registro:
                registros.append(registro)
            bloque = []
        elif linea:
            bloque.append(linea)
    return registros


def _leer_bloque(bloque: list[str]) -> dict | None:
    # Se descartan las líneas «Lectura rápida - dominio», que imitan una línea
    # de autores. Primero se busca la línea de autores con año; solo si no hay
    # ninguna se acepta una sin año.
    bloque = [linea for linea in bloque if not LECTURA_RAPIDA.match(linea)]
    for patron in (AUTORES, AUTORES_SIN_ANIO):
        for i, linea in enumerate(bloque):
            autores = patron.match(linea)
            if autores and i > 0:
                return _registro(bloque, i, autores)
    return None


def _registro(bloque: list[str], i: int, autores: re.Match) -> dict:
    """Arma el registro a partir de la línea de autores en la posición i."""
    return {
        "año": int(autores["anio"]) if "anio" in autores.groupdict() else None,
        "título": PREFIJO.sub("", bloque[i - 1]).strip(),
        "autores": autores["autores"].strip(),
        "fuente": autores["dominio"],
        "fragmento": " ".join(bloque[i + 1:]),
    }


def leer_google_scholar(carpeta: Path, consultas: dict[str, str] | None = None):
    """Lee ``<cadena>.txt`` de ``carpeta`` y devuelve (búsquedas, registros).

    La fecha de cada búsqueda es la de modificación del archivo. Los
    registros repetidos dentro de una misma cadena (una página pegada dos
    veces) se cuentan una sola vez.
    """
    busquedas, registros = [], []
    for ruta in sorted(Path(carpeta).glob("*.txt")):
        cadena = ruta.stem
        lineas = ruta.read_text(encoding="utf-8-sig").splitlines()
        fecha = date.fromtimestamp(ruta.stat().st_mtime).isoformat()
        encontrados = pd.DataFrame(_registros(lineas))
        if not encontrados.empty:
            encontrados = encontrados.drop_duplicates("título")
        busquedas.append({"fecha": fecha, "base": "Google Scholar", "cadena": cadena,
                          "consulta": (consultas or {}).get(cadena, next((l.strip() for l in lineas if l.strip()), "")),
                          "resultados": _total(lineas)})
        for fila in encontrados.to_dict("records"):
            registros.append({"fecha": fecha, "base": "Google Scholar", "cadena": cadena,
                              "año": fila["año"], "título": fila["título"],
                              "enlace": fila["fuente"], "autores": fila["autores"],
                              "fragmento": fila["fragmento"]})
    return pd.DataFrame(busquedas), pd.DataFrame(registros)
