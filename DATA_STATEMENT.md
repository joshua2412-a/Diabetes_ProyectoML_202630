# Declaración de los datos

## Conjunto principal: Diabetes 130-US Hospitals for Years 1999-2008

| Campo | Valor |
|---|---|
| Fuente | UCI Machine Learning Repository, conjunto 296 (DOI 10.24432/C5230J) |
| Creadores (según UCI) | John Clore, Krzysztof Cios, Jon DeShazo y Beata Strack |
| Artículo de origen | Strack et al. (2014), *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*, BioMed Research International |
| Procedencia | Base de datos Health Facts (Cerner Corporation): registros clínicos y administrativos de 130 hospitales de Estados Unidos |
| Periodo | 1999 a 2008 |
| Licencia | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Tamaño original | 101 766 encuentros de 71 518 pacientes, 50 columnas |
| Tras el ETL | 99 340 encuentros de 69 987 pacientes (capítulo 2, `results/trazabilidad_etl.csv`) |
| Objetivo | `readmitted_30 = 1` si el paciente se readmite antes de 30 días (11.39 %) |

**Criterios de inclusión del conjunto original:**
- encuentros de hospitalización de pacientes con cualquier diagnóstico de diabetes;
- estancia de 1 a 14 días;
- con pruebas de laboratorio y medicación durante la estancia.

**Decisiones del ETL** (capítulo 2):
- **Exclusiones:** se excluyen las altas por fallecimiento u hospicio, porque no pueden readmitirse, y los registros con género inválido.
- **Unidad de análisis:** se conservan todos los encuentros de cada paciente. Elegir uno por paciente sesgaba la tasa de readmisión. Por eso todas las particiones se agrupan por paciente.
- **Variables eliminadas:** `weight` (97 % de faltantes) y `payer_code` (identifica al asegurador).
- **Faltantes como categoría:** en `race`, `medical_specialty` y los diagnósticos, el faltante se trata como una categoría explícita.
- **Diagnósticos:** los códigos ICD-9 se agrupan en capítulos.

**Sesgos y limitaciones conocidos:**
- **Antigüedad:** los datos tienen más de 15 años, y la práctica clínica, la codificación ICD-9 (hoy ICD-10) y las políticas de readmisión han cambiado.
- **Cobertura:** solo incluye hospitales de Estados Unidos que usaban el sistema de Cerner. No es una muestra representativa de la población diabética.
- **Readmisiones no observadas:** se ven solo las que ocurren dentro de la misma red de hospitales. Un paciente readmitido en otra institución figura como no readmitido.
- **Raza:** la población es mayoritariamente caucásica (75 %), y el 2.2 % de los encuentros no tiene raza registrada. Los grupos pequeños (Asian, Hispanic, Other) tienen estimaciones de desempeño más inciertas.
- **Datos sensibles:** `race`, `gender` y `age` son atributos protegidos; su uso se discute en `MODEL_CARD.md`.
- **Privacidad:** los datos están anonimizados, sin nombres ni fechas exactas. Los identificadores `encounter_id` y `patient_nbr` son seudónimos y solo se usan para agrupar las particiones.

## Conjuntos adicionales (evaluación ampliada, capítulo 16)

| Conjunto | Fuente | Licencia | Uso en el proyecto |
|---|---|---|---|
| Default of Credit Card Clients (Yeh y Lien, 2009) | UCI 350, DOI 10.24432/C55S3H | CC BY 4.0 | 30 000 clientes de Taiwán (2005); 22.1 % de impago |
| Bank Marketing (Moro, Rita y Cortez, 2014) | UCI 222, DOI 10.24432/C5K306 | CC BY 4.0 | 45 211 llamadas de un banco portugués (2008-2010); 11.7 % de suscripción. Se elimina `duration`, que se conoce solo después de la llamada, y las filas se barajan porque el archivo está ordenado por fecha |
| CDC Diabetes Health Indicators | UCI 891, DOI 10.24432/C53919 | UCI no fija una licencia propia y remite al conjunto enlazado, derivado de la encuesta BRFSS 2015 de los CDC | Submuestra estratificada de 100 000 de 253 680 respuestas; 13.9 % con diabetes o prediabetes |

Los conjuntos adicionales se descargan con `experiments/conjuntos_adicionales.py` y no se versionan en el repositorio (`data/raw/adicionales/` está en `.gitignore`).
