# Predicción de readmisión hospitalaria en pacientes diabéticos

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24211425.svg)](https://doi.org/10.5281/zenodo.24211425)

Proyecto final del curso de Machine Learning (Universidad del Norte, 2026) de **Joshua Hincapié Llorente** y **Alejandro Cantillo Escorcia**.

Con los datos de *Diabetes 130-US Hospitals* (UCI, 1999-2008), el proyecto predice qué pacientes vuelven al hospital antes de 30 días del alta. Hace tres cosas:
- compara 7 modelos con 4 estrategias de balanceo y 4 optimizadores, más LightGBM y CatBoost;
- propone e implementa un modelo original, **DES-CPL**: selección dinámica de ensambles por competencia probabilística local;
- valida todo con un protocolo pre-registrado.

| Pieza | Dónde |
|---|---|
| Libro (Jupyter Book) | `docs/`, publicado en https://joshua2412-a.github.io/Diabetes_ProyectoML_202630 |
| Dashboard (Dash) | `app.py`, `pages/`, `src/tablero/`; desplegado en Render: *URL pendiente del despliegue* |
| Archivo versionado (Zenodo) | DOI [10.5281/zenodo.24211425](https://doi.org/10.5281/zenodo.24211425) |
| Modelo original | `src/original_model.py` (interfaz) y `src/des_cpl.py` (implementación) |
| Protocolo pre-registrado | `experiments/protocolo.md`, etiqueta de Git `protocolo-v1` |
| Experimentos | `experiments/` (scripts) y `results/` (resultados en CSV) |
| Modelos serializados | `artifacts/` (joblib + `metadata.json`) |
| Pruebas | `tests/` (pytest; se ejecutan en GitHub Actions) |
| Ficha del modelo y declaración de los datos | `MODEL_CARD.md`, `DATA_STATEMENT.md` |

## Resultado principal

Los mejores modelos empatan en AUC-PR, alrededor de 0.23 con una prevalencia del 11.4 %:
- en el bucle externo, XGBoost 0.2290 y DES-CPL 0.2279;
- en prueba, XGBoost 0.2315 y DES-CPL 0.2293.

DES-CPL es original (capítulo 14) y está implementado por los autores, pero **no supera de forma significativa al mejor benchmark** en ninguna de las tres pruebas del protocolo (capítulos 13 y 16). Sí resulta:
- más estable que XGBoost frente a la semilla y a perturbaciones de los datos;
- por delante de todos los modelos individuales en rango medio, en cuatro conjuntos de datos (capítulo 16).

## Instalación

Requiere Python 3.12.13. Con conda:

```bash
conda create -n ml_proyecto python=3.12.13
conda activate ml_proyecto
pip install -r requirements.txt
```

`requirements.txt` fija las versiones exactas. El dashboard en producción usa el subconjunto de `requirements-render.txt`.

## Uso

Todos los comandos se ejecutan desde la raíz del repositorio.

```bash
# Pruebas rápidas (las mismas de la integración continua)
pytest -m "not lento"

# Dashboard en local: http://127.0.0.1:8050
python app.py

# Modelo original con la interfaz de scikit-learn
python -c "from src.original_model import ModeloOriginal; help(ModeloOriginal)"
```

**Notebooks.** Se ejecutan desde `docs/`, por ejemplo `jupyter nbconvert --to notebook --execute --inplace 16_Evaluacion_Ampliada.ipynb`. Leen sus resultados de `results/` y solo calculan lo que falta.

**Experimentos largos.** Se corren con los scripts de `experiments/`. Todos guardan puntos de control y continúan donde quedaron si se interrumpen.

| Script | Qué hace | Tiempo aproximado* |
|---|---|---|
| `des_cpl.py` | DES-CPL en los 5 folds externos (capítulo 13) | 1 h |
| `benchmarks_sota.py` | LightGBM y CatBoost (capítulo 13) | 4 h |
| `evaluacion_prueba.py` | Evaluación final en prueba (sección 13.8) | 2 h |
| `conjuntos_adicionales.py` | Tres conjuntos adicionales (capítulo 16); `--prueba` hace un ensayo de minutos | 5 h |
| `semillas_robustez.py` | Sensibilidad, robustez y 10 semillas (capítulo 16) | 8 h |
| `artefacto_des_cpl.py` | Artefacto de DES-CPL, SHAP y tabla de presupuesto | 20 min |
| `revision_literatura.py` | Búsqueda de literatura por API (capítulo 14) | 2 min |
| `tabla_maestra_unificada.py` | Tabla maestra de todos los experimentos | segundos |

\* En el equipo de los autores: 4 núcleos físicos, `N_JOBS = 4`.

**Libro.**

```bash
jupyter-book build docs/
ghp-import -n -p -f docs/_build/html
```

## Reproducibilidad

- **Semilla global:** 42 (`src/configuracion.py`), propagada a numpy, scikit-learn, XGBoost, Optuna y DEAP.
- **Particiones:** agrupadas por paciente (`patient_nbr`) en todos los niveles.
- **Lo que hay que reejecutar:** los notebooks no se reejecutan al construir el libro (`execute_notebooks: off`). Si cambia un resultado, hay que volver a ejecutar su notebook.
- **Datos:**
  - `data/raw/diabetic_data.csv` se descarga de UCI (DOI 10.24432/C5230J);
  - el capítulo 2 genera `data/processed/`;
  - los conjuntos adicionales los descarga `experiments/conjuntos_adicionales.py`.

## Despliegue

`render.yaml` define un servicio web gratuito en Render: `gunicorn app:server --workers 1 --threads 4`. El tablero no entrena nada; solo carga los artefactos de `artifacts/`. Los pasos:
1. Conectar el repositorio a Render como *Blueprint*.
2. Esperar la construcción.
3. Abrir la URL pública varios minutos antes de usarla: el plan gratuito duerme el servicio cuando está inactivo.

## Licencia y cita

- **Código:** licencia MIT (`LICENSE`).
- **Datos:** conservan sus propias licencias (`DATA_STATEMENT.md`).
- **Cómo citar:** ver `CITATION.cff`.
