"""
Búsqueda sistemática de literatura para la declaración de novedad (capítulo 14).

Ejecuta las ocho cadenas de búsqueda del protocolo en cuatro bases con API
pública y guarda, con la fecha de la consulta:

- results/literatura/busquedas_<fecha>.csv  (base, cadena, consulta, resultados)
- results/literatura/registros_<fecha>.csv  (base, cadena, año, título, enlace)

Bases:
- OpenAlex (índice multidisciplinario; filtro en título y resumen)
- PubMed (E-utilities; campo [tiab])
- arXiv (API de exportación; campo abs)
- Semantic Scholar (búsqueda masiva; título y resumen)

La guía pide repetir la búsqueda antes de la entrega final. Para hacerlo
basta con volver a correr este script: crea archivos nuevos con la fecha del
día y no borra los anteriores, de modo que el capítulo 14 puede comparar
ambas consultas.

Uso (raíz del repositorio; solo usa la biblioteca estándar):

    python experiments/revision_literatura.py

Semantic Scholar limita las consultas sin clave; si responde 429, el script
espera y reintenta.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / "results" / "literatura"
MAXIMO_REGISTROS = 100

# Cada cadena combina (i) la familia del modelo, (ii) el problema y
# (iii) sinónimos; C5 es la versión en español. La sintaxis cambia por base,
# pero los términos son los mismos.
CADENAS = {
    "C1": "Selección dinámica de ensambles + readmisión",
    "C2": "Selección dinámica de clasificadores + readmisión + diabetes",
    "C3": "Métodos de selección dinámica (KNORA, META-DES) + diabetes",
    "C4": "Conjunto Diabetes 130-US + ensambles o selección dinámica",
    "C5": "Cadena en español: selección dinámica + readmisión o diabetes",
    "C6": "Región o competencia local + readmisión o diabetes",
    "C7": "Componentes de DES-CPL: encogimiento, competencia probabilística, perfil de salidas",
    "C8": "Cualquier selección dinámica + readmisión (sin restringir enfermedad)",
}

OPENALEX = {
    "C1": '"dynamic ensemble selection" AND (readmission OR rehospitalization OR readmitted)',
    "C2": '("dynamic classifier selection" OR "dynamic selection") AND (readmission OR rehospitalization) AND (diabetes OR diabetic)',
    "C3": '("dynamic ensemble selection" OR "dynamic classifier selection" OR KNORA OR "META-DES") AND (diabetes OR diabetic)',
    "C4": '("130-US hospitals" OR "130 US hospitals" OR "Diabetes 130") AND (ensemble OR "dynamic selection")',
    "C5": '("selección dinámica" OR "seleccion dinamica") AND (clasificadores OR ensambles OR ensamble) AND (readmisión OR reingreso OR diabetes)',
    "C6": '("local competence" OR "region of competence" OR "competence region") AND (readmission OR diabetes OR diabetic)',
    "C7": '"dynamic ensemble selection" AND (shrinkage OR "probabilistic competence" OR "output profile" OR "output profiles")',
    "C8": '("dynamic ensemble selection" OR "dynamic classifier selection" OR "dynamic selection") AND (readmission OR rehospitalization OR readmitted)',
}

PUBMED = {
    "C1": '"dynamic ensemble selection"[tiab] AND (readmission*[tiab] OR rehospitali*[tiab] OR readmitted[tiab])',
    "C2": '("dynamic classifier selection"[tiab] OR "dynamic selection"[tiab]) AND (readmission*[tiab] OR rehospitali*[tiab]) AND diabet*[tiab]',
    "C3": '("dynamic ensemble selection"[tiab] OR "dynamic classifier selection"[tiab] OR KNORA[tiab] OR "META-DES"[tiab]) AND diabet*[tiab]',
    "C4": '("130-US hospitals"[tiab] OR "130 US hospitals"[tiab] OR "Diabetes 130"[tiab]) AND (ensemble*[tiab] OR "dynamic selection"[tiab])',
    "C5": '("selección dinámica"[tiab] OR "seleccion dinamica"[tiab]) AND (clasificador*[tiab] OR ensamble*[tiab]) AND (readmision[tiab] OR readmisión[tiab] OR reingreso*[tiab] OR diabetes[tiab])',
    "C6": '("local competence"[tiab] OR "region of competence"[tiab] OR "competence region"[tiab]) AND (readmission*[tiab] OR diabet*[tiab])',
    "C7": '"dynamic ensemble selection"[tiab] AND (shrinkage[tiab] OR "probabilistic competence"[tiab] OR "output profile*"[tiab])',
    "C8": '("dynamic ensemble selection"[tiab] OR "dynamic classifier selection"[tiab] OR "dynamic selection"[tiab]) AND (readmission*[tiab] OR rehospitali*[tiab])',
}

ARXIV = {
    "C1": 'abs:"dynamic ensemble selection" AND (abs:readmission OR abs:rehospitalization)',
    "C2": '(abs:"dynamic classifier selection" OR abs:"dynamic selection") AND abs:readmission AND abs:diabetes',
    "C3": '(abs:"dynamic ensemble selection" OR abs:"dynamic classifier selection" OR abs:KNORA OR abs:"META-DES") AND (abs:diabetes OR abs:diabetic)',
    "C4": '(abs:"130-US hospitals" OR abs:"130 US hospitals" OR abs:"Diabetes 130") AND (abs:ensemble OR abs:"dynamic selection")',
    "C5": 'abs:"seleccion dinamica" AND (abs:readmision OR abs:diabetes)',
    "C6": '(abs:"local competence" OR abs:"region of competence" OR abs:"competence region") AND (abs:readmission OR abs:diabetes)',
    "C7": 'abs:"dynamic ensemble selection" AND (abs:shrinkage OR abs:"probabilistic competence" OR abs:"output profile")',
    "C8": '(abs:"dynamic ensemble selection" OR abs:"dynamic classifier selection" OR abs:"dynamic selection") AND (abs:readmission OR abs:rehospitalization)',
}

# En la búsqueda masiva de Semantic Scholar, «+» es AND y «|» es OR.
SEMANTIC_SCHOLAR = {
    "C1": '"dynamic ensemble selection" + (readmission | rehospitalization | readmitted)',
    "C2": '("dynamic classifier selection" | "dynamic selection") + (readmission | rehospitalization) + (diabetes | diabetic)',
    "C3": '("dynamic ensemble selection" | "dynamic classifier selection" | KNORA | "META-DES") + (diabetes | diabetic)',
    "C4": '("130-US hospitals" | "130 US hospitals" | "Diabetes 130") + (ensemble | "dynamic selection")',
    "C5": '("selección dinámica" | "seleccion dinamica") + (clasificadores | ensambles) + (readmisión | reingreso | diabetes)',
    "C6": '("local competence" | "region of competence" | "competence region") + (readmission | diabetes | diabetic)',
    "C7": '"dynamic ensemble selection" + (shrinkage | "probabilistic competence" | "output profile")',
    "C8": '("dynamic ensemble selection" | "dynamic classifier selection" | "dynamic selection") + (readmission | rehospitalization)',
}


def leer(url: str, intentos: int = 5, espera: float = 6.0) -> bytes:
    """Descarga una URL; reintenta si la base limita la frecuencia (429)."""
    peticion = urllib.request.Request(url, headers={"User-Agent": "revision-literatura/1.0"})
    for intento in range(intentos):
        try:
            with urllib.request.urlopen(peticion, timeout=60) as respuesta:
                return respuesta.read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or intento == intentos - 1:
                raise
            time.sleep(espera * (intento + 1))
    raise RuntimeError(url)


def openalex(consulta: str):
    url = ("https://api.openalex.org/works?per-page=100&select=id,display_name,publication_year,doi"
           "&filter=title_and_abstract.search:" + urllib.parse.quote(consulta))
    datos = json.loads(leer(url))
    registros = [(r["publication_year"], r["display_name"], r["doi"] or r["id"])
                 for r in datos["results"]]
    return datos["meta"]["count"], registros


def pubmed(consulta: str):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    busqueda = json.loads(leer(f"{base}esearch.fcgi?db=pubmed&retmode=json&retmax={MAXIMO_REGISTROS}"
                               f"&term={urllib.parse.quote(consulta)}"))["esearchresult"]
    ids = busqueda["idlist"]
    registros = []
    if ids:
        resumen = json.loads(leer(f"{base}esummary.fcgi?db=pubmed&retmode=json&id={','.join(ids)}"))
        for i in ids:
            r = resumen["result"][i]
            doi = next((a["value"] for a in r.get("articleids", []) if a["idtype"] == "doi"), None)
            registros.append((r.get("pubdate", "")[:4], r["title"],
                              f"https://doi.org/{doi}" if doi else f"PMID:{i}"))
    time.sleep(0.4)          # PubMed admite tres consultas por segundo sin clave
    return int(busqueda["count"]), registros


def arxiv(consulta: str):
    url = ("https://export.arxiv.org/api/query?max_results=100&search_query="
           + urllib.parse.quote(consulta))
    raiz = ET.fromstring(leer(url))
    ns = {"a": "http://www.w3.org/2005/Atom",
          "o": "http://a9.com/-/spec/opensearch/1.1/"}
    total = int(raiz.find("o:totalResults", ns).text)
    registros = [(e.find("a:published", ns).text[:4],
                  " ".join(e.find("a:title", ns).text.split()),
                  e.find("a:id", ns).text) for e in raiz.findall("a:entry", ns)]
    time.sleep(3.1)          # arXiv pide al menos 3 s entre consultas
    return total, registros


def semantic_scholar(consulta: str):
    url = ("https://api.semanticscholar.org/graph/v1/paper/search/bulk?fields=title,year,externalIds"
           "&query=" + urllib.parse.quote(consulta))
    datos = json.loads(leer(url))
    registros = []
    for r in datos.get("data", [])[:MAXIMO_REGISTROS]:
        doi = (r.get("externalIds") or {}).get("DOI")
        registros.append((r.get("year"), r["title"],
                          f"https://doi.org/{doi}" if doi else f"S2:{r['paperId']}"))
    time.sleep(3.0)
    return datos.get("total", 0), registros


BASES = {
    "OpenAlex": (openalex, OPENALEX),
    "PubMed": (pubmed, PUBMED),
    "arXiv": (arxiv, ARXIV),
    "Semantic Scholar": (semantic_scholar, SEMANTIC_SCHOLAR),
}


def main():
    hoy = date.today().isoformat()
    CARPETA.mkdir(parents=True, exist_ok=True)
    busquedas, registros = [], []
    for base, (consultar, cadenas) in BASES.items():
        for clave, consulta in cadenas.items():
            try:
                total, encontrados = consultar(consulta)
            except Exception as error:                     # la base no respondió
                print(f"[error] {base} {clave}: {error}")
                total, encontrados = None, []
            busquedas.append({"fecha": hoy, "base": base, "cadena": clave,
                              "descripción": CADENAS[clave], "consulta": consulta,
                              "resultados": total})
            registros += [{"fecha": hoy, "base": base, "cadena": clave, "año": a,
                           "título": t, "enlace": e} for a, t, e in encontrados]
            print(f"{base:17s} {clave}: {total}")

    for nombre, filas in (("busquedas", busquedas), ("registros", registros)):
        ruta = CARPETA / f"{nombre}_{hoy}.csv"
        campos = list(filas[0]) if filas else ["fecha"]
        with ruta.open("w", newline="", encoding="utf-8") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(filas)
        print(f"Guardado {ruta.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
