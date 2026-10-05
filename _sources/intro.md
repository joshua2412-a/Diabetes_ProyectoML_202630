<div align="center">

# Predicción de Readmisión Hospitalaria en Pacientes Diabéticos

**Optimización y Comparación Estadística de Modelos de Aprendizaje Automático sobre Registros Clínicos de 130 Hospitales de EE. UU.**
<br>

**Autores**

Alejandro Cantillo Escorcia

Joshua Hincapie LLorente

<br>

---

</div>

La diabetes mellitus es una de las enfermedades crónicas con mayor carga sobre los sistemas de salud, y los pacientes diabéticos hospitalizados presentan tasas de readmisión elevadas. Las readmisiones tempranas, en particular las que ocurren dentro de los 30 días posteriores al alta, se consideran un indicador de la calidad de la atención y generan costos considerables tanto para las instituciones como para los pacientes.

Este proyecto aborda este desafío mediante la integración de un proceso ETL, Análisis Exploratorio de Datos (EDA) y un pipeline reproducible de aprendizaje automático que combina distintos modelos, técnicas de balanceo de clases y métodos de optimización de hiperparámetros. El objetivo central es identificar, con rigor estadístico, qué modelos y factores clínicos permiten anticipar la readmisión temprana de pacientes diabéticos, apoyando la toma de decisiones clínicas al momento del alta.

**Ficha Técnica del Dataset**

- **Nombre:** Diabetes 130-US Hospitals for Years 1999-2008
- **Fuente:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Muestra:** 101,766 encuentros hospitalarios de 71,518 pacientes.
- **Atributos (50):** variables demográficas, administrativas, diagnósticos (ICD-9), resultados de laboratorio y 23 medicamentos para la diabetes.
- **Variable Objetivo:** readmisión hospitalaria (`readmitted`: `<30`, `>30`, `NO`), modelada como clasificación binaria entre readmisión en menos de 30 días y el resto.
- **Naturaleza de datos:** Tabular, multivariada (numérica y categórica).

```{tableofcontents}
```
