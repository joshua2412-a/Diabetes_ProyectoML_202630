<div align="center">

# Predicción de Readmisión Hospitalaria en Pacientes Diabéticos

**Optimización y Comparación Estadística de Modelos de Aprendizaje Automático sobre Registros Clínicos de 130 Hospitales de EE. UU.**
<br>

**Autores**

Alejandro Cantillo Escorcia

Joshua Hincapié Llorente

<br>

**Dashboard:** [https://readmision-diabetes.onrender.com](https://readmision-diabetes.onrender.com) · **Código:** [GitHub](https://github.com/joshua2412-a/Diabetes_ProyectoML_202630) · **Archivo:** DOI [10.5281/zenodo.24211424](https://doi.org/10.5281/zenodo.24211424)

---

</div>

La diabetes mellitus es una de las enfermedades crónicas con mayor carga sobre los sistemas de salud, y los pacientes diabéticos hospitalizados presentan tasas de readmisión elevadas. Las readmisiones tempranas, en particular las que ocurren dentro de los 30 días posteriores al alta, se consideran un indicador de la calidad de la atención y generan costos considerables tanto para las instituciones como para los pacientes.

Un equipo de seguimiento tiene capacidad limitada: no puede llamar ni citar a todos los pacientes que salen del hospital. Ordenarlos por su riesgo de readmisión permite concentrar ese esfuerzo donde más sirve. El proyecto aborda ese problema con un proceso ETL, un análisis exploratorio y un pipeline reproducible de aprendizaje automático, y evalúa con rigor estadístico hasta dónde llegan los modelos con la información disponible al alta.

## Resumen ejecutivo

Este libro predice qué pacientes diabéticos vuelven al hospital antes de 30 días del alta, con los registros de *Diabetes 130-US Hospitals* (99 340 encuentros de 69 987 pacientes tras el ETL; 11.4 % de readmisiones tempranas).

- **Benchmark.** Se compararon 7 modelos con 4 estrategias de balanceo y 4 optimizadores de hiperparámetros (112 combinaciones), con validación anidada agrupada por paciente, más LightGBM y CatBoost. Los mejores empatan: XGBoost alcanza un AUC-PR de 0.2290 en el bucle externo y ninguna prueba lo separa de Random Forest, CatBoost o LightGBM.
- **Modelo original.** Se propone e implementa **DES-CPL**, una selección dinámica de ensambles por competencia probabilística local. Obtiene 0.2279 en el bucle externo y 0.2293 en prueba, frente a 0.2315 de XGBoost. Es nuevo en el problema y está programado por los autores, pero no supera de forma significativa al mejor benchmark en ninguna de las tres pruebas del protocolo pre-registrado. Sí es más estable frente a la semilla y a perturbaciones de los datos.
- **Límite de los datos.** Con las 38 variables disponibles, el AUC-PR tiene un techo cercano a 0.23 (AUC-ROC de 0.67) que no rompen ni más presupuesto de búsqueda, ni otra familia de modelos, ni combinarlos.
- **Producto.** Los modelos están serializados y se usan en un dashboard público con predicción en vivo, explicación local e incertidumbre.

## Objetivos

**Objetivo general.** Identificar, con rigor estadístico, qué modelos y qué factores clínicos permiten anticipar la readmisión temprana de pacientes diabéticos, y poner los mejores modelos a disposición en una herramienta de apoyo a la decisión al momento del alta.

**Objetivos específicos.**

1. Depurar y explorar los datos, y fijar una validación que no filtre información entre encuentros del mismo paciente.
2. Comparar modelos, estrategias de balanceo y optimizadores de hiperparámetros con validación anidada y pruebas estadísticas.
3. Diseñar, implementar y validar un modelo original bajo un protocolo pre-registrado, con conjuntos de datos adicionales y estudios de robustez.
4. Desplegar los modelos en un dashboard público y dejar el proyecto reproducible.

## Contribuciones

1. **Un benchmark completo y auditado.** 112 combinaciones de modelo, balanceo y optimizador con validación anidada 5 × 3 agrupada por paciente, y la medición del sesgo que introduce validar por filas (capítulos 6 a 11).
2. **Las correcciones del Entregable 2 con evidencia.** Un estudio de convergencia muestra que el presupuesto de búsqueda original era suficiente; se añaden la d de Cohen, una auditoría de fuga de información, pruebas automáticas y la división del código en módulos (capítulo 12).
3. **Un modelo original, DES-CPL.** Combina cuatro mecanismos: competencia medida con el puntaje de Brier reponderado por la prevalencia local, región de competencia estratificada en el espacio de logits que excluye al propio paciente, encogimiento de la competencia local hacia la global y poda previa. Tiene formulación, cinco proposiciones con sus demostraciones y análisis de complejidad (capítulos 13 y 15).
4. **Evidencia documentada de novedad.** Una revisión en cinco bases bibliográficas con ocho cadenas de búsqueda, búsqueda por citación y repositorios de código, repetida antes de cerrar el trabajo (capítulo 14).
5. **Una evaluación pre-registrada y un resultado negativo reportado como tal.** Tres pruebas de hipótesis fijadas antes de los experimentos definitivos, tres conjuntos de datos adicionales, diez semillas y estudios de robustez, sensibilidad, calibración e incertidumbre (capítulos 13 y 16).
6. **Un diagnóstico del techo de desempeño.** La diferencia de pérdida entre modelos no se puede predecir a partir del paciente, lo que explica por qué ninguna combinación local supera a una global (capítulos 5 y 13).
7. **Un producto reproducible.** Dashboard desplegado, modelos serializados con sus metadatos, ficha del modelo, declaración de los datos y archivo con DOI (capítulos 18 y 19).

## Ficha técnica del conjunto de datos

- **Nombre:** Diabetes 130-US Hospitals for Years 1999-2008
- **Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Muestra:** 101 766 encuentros hospitalarios de 71 518 pacientes.
- **Atributos (50):** variables demográficas, administrativas, diagnósticos (ICD-9), resultados de laboratorio y 23 medicamentos para la diabetes.
- **Variable Objetivo:** readmisión hospitalaria (`readmitted`: `<30`, `>30`, `NO`), modelada como clasificación binaria entre readmisión en menos de 30 días y el resto.
- **Naturaleza de datos:** Tabular, multivariada (numérica y categórica).

## Mapa del libro

El libro conserva el orden en que se hizo el trabajo: los capítulos 1 a 5 cubren los datos y su exploración; los capítulos 6 a 11, el experimento de comparación de modelos; los capítulos 12 a 19, la revisión del experimento, el modelo original y el producto; y los capítulos 20 y 21, las comprobaciones posteriores al protocolo.

| Capítulos | Contenido |
|---|---|
| [1](01_Datos.ipynb) a [5](05_Sintesis.ipynb) | Datos, ETL, análisis univariado y bivariado, y síntesis del análisis exploratorio |
| [6](06_Preprocesamiento.ipynb) y [7](07_ModeloBase.ipynb) | Preprocesamiento, validación agrupada por paciente y modelo base |
| [8](08_Clasificacion.ipynb) a [11](11_Comparacion_Estadistica.ipynb) | Experimento combinatorio, optimizadores, evaluación del modelo final y comparación estadística |
| [12](12_Correcciones.ipynb) | Correcciones del Entregable 2 y tabla de trazabilidad |
| [13](13_Modelo_Original.ipynb) | Modelo original: motivación, diseño, resultados definitivos y evaluación en prueba |
| [14](14_Revision_Literatura.ipynb) | Revisión de literatura y declaración de novedad |
| [15](15_Formulacion_DES_CPL.ipynb) | Formulación, pseudocódigo, propiedades y complejidad de DES-CPL |
| [16](16_Evaluacion_Ampliada.ipynb) | Conjuntos adicionales, robustez, semillas, sensibilidad, interpretabilidad, equidad y costo |
| [17](17_Conclusiones.ipynb) | Conclusiones, veredicto, limitaciones y declaración de autoría |
| [18](18_Reproducibilidad.ipynb) | Reproducibilidad, tabla maestra de experimentos y bibliografía |
| [19](19_Artefactos_Dashboard.ipynb) | Mejores combinaciones, artefactos serializados, dashboard, despliegue y ficha del modelo |
| [20](20_Verificaciones.ipynb) | Verificaciones posteriores al protocolo: ¿cambia algún resultado si se corrigen las decisiones objetables? |
| [21](21_Exploracion_Mejoras.ipynb) | Exploración de mejoras de DES-CPL y resumen de todas las comprobaciones posteriores |

**Correspondencia con la estructura de la guía del tercer entregable.**

| Capítulo de la guía | Dónde está en este libro |
|---|---|
| 0. Introducción | Esta página |
| 1. Trazabilidad de correcciones | Capítulo [12](12_Correcciones.ipynb) (tabla en la sección 12.6) |
| 2. Datos: ETL y EDA | Capítulos [1](01_Datos.ipynb) a [5](05_Sintesis.ipynb); auditoría de fuga en la sección 12.4; declaración de los datos en `DATA_STATEMENT.md` |
| 3. Modelos benchmark | Capítulos [6](06_Preprocesamiento.ipynb) a [11](11_Comparacion_Estadistica.ipynb); presupuesto de 60 evaluaciones en la sección 12.2; LightGBM y CatBoost en la sección 13.5; mejores combinaciones y artefactos en las secciones 19.1 a 19.5 |
| 4. Revisión de literatura y novedad | Capítulo [14](14_Revision_Literatura.ipynb) |
| 5. Modelo original | Capítulos [13](13_Modelo_Original.ipynb) (motivación, diseño y ablación) y [15](15_Formulacion_DES_CPL.ipynb) (formulación, pseudocódigo, sustento teórico y complejidad) |
| 6. Protocolo y validación estadística | `experiments/protocolo.md` (etiqueta `protocolo-v1`, 9 de noviembre de 2026); secciones 13.5, 13.8 y 16.9; comprobaciones posteriores en los capítulos [20](20_Verificaciones.ipynb) y [21](21_Exploracion_Mejoras.ipynb) |
| 7. Evaluación ampliada y robustez | Capítulo [16](16_Evaluacion_Ampliada.ipynb); calibración e incertidumbre en la sección 13.8 |
| 8. Dashboard y despliegue | Capítulo [19](19_Artefactos_Dashboard.ipynb) (secciones 19.6 a 19.8); [dashboard público](https://readmision-diabetes.onrender.com) |
| 9. Conclusiones y limitaciones | Capítulo [17](17_Conclusiones.ipynb) |
| 10. Reproducibilidad y bibliografía | Capítulo [18](18_Reproducibilidad.ipynb) |

```{tableofcontents}
```
