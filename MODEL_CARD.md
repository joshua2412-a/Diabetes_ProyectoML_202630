# Ficha del modelo: DES-CPL

**Versión:** 1.0, noviembre de 2026.
**Autores:** Joshua Hincapié Llorente y Alejandro Cantillo Escorcia (Universidad del Norte).
**Artefacto:** `artifacts/des_cpl.joblib`. Sus metadatos están en `artifacts/des_cpl.metadata.json`.

## Descripción

DES-CPL (*selección dinámica de ensambles por competencia probabilística local*) combina cuatro clasificadores: árbol de decisión, Random Forest, XGBoost y SVM lineal calibrado. Para cada paciente, decide cuánto confiar en cada modelo según la calidad de sus probabilidades (puntuación de Brier) en pacientes con un perfil de predicciones parecido. Esa competencia local se encoge hacia la global cuando es ruidosa.

**Configuración del artefacto:** $k = 20$ vecinos por clase, $\tau = 0.0003$ y $\lambda = 0.9$, elegidos con 60 evaluaciones de Optuna sobre el conjunto de selección. La formulación y sus propiedades están en el capítulo 15 del libro.

## Uso previsto

- **Para qué:** apoyar la priorización del seguimiento posterior al alta. Ordena a los pacientes diabéticos hospitalizados según su riesgo de readmisión antes de 30 días, para que un equipo con capacidad limitada concentre llamadas o citas en el 15 % de mayor riesgo, que es el umbral operativo del artefacto.
- **Usuarios:** analistas y equipos de gestión hospitalaria, como herramienta de **apoyo**. La decisión clínica es del personal de salud.
- **Alcance:** es un proyecto académico. No está validado para uso clínico.

## Datos de entrenamiento

- **Datos:** *Diabetes 130-US Hospitals for Years 1999-2008* (UCI). Detalles en `DATA_STATEMENT.md`.
- **Unidad:** el encuentro hospitalario.
- **Tamaño:** 79 473 encuentros de entrenamiento (56 036 pacientes) y 19 867 de prueba (13 951), sin pacientes compartidos.
- **Etiqueta:** readmisión antes de 30 días, con una prevalencia del 11.4 %.
- **Variables:** 38 predictores administrativos y clínicos disponibles al alta, como demografía, diagnósticos agrupados, número de visitas previas y medicamentos.

## Métricas

| Evaluación | AUC-PR | AUC-ROC | Brier |
|---|---|---|---|
| Bucle externo (5 folds agrupados por paciente) | 0.2279 ± 0.0174 | 0.6721 | 0.0963 |
| Prueba (una sola vez) | 0.2293 | 0.6661 | 0.0961 |
| Mejor benchmark en prueba (XGBoost) | 0.2315 | 0.6664 | 0.0960 |

- **Calibración:** el error de calibración esperado en prueba es 0.0065.
- **Predicción conforme (α = 0.1):** la cobertura es 0.890 en total y 0.869 en los readmitidos.
- **Variabilidad:** con 10 semillas, el AUC-PR es 0.2270 ± 0.0007. El de XGBoost es 0.2251 ± 0.0017.
- **Robustez:** con ruido en las etiquetas, menos prevalencia o menos datos, DES-CPL pierde entre 1.6 y 1.9 veces menos AUC-PR que XGBoost (capítulo 16).

## Limitaciones

- **No supera al mejor modelo individual.** Las diferencias con XGBoost, LightGBM y CatBoost no son significativas (capítulos 13 y 16).
- **Capacidad de discriminación modesta.** Con un AUC-ROC de 0.67, en el 15 % de mayor riesgo solo uno de cada cuatro o cinco pacientes marcados se readmite en realidad. Para la mayoría de los pacientes, el conjunto de predicción conforme contiene las dos clases.
- **Datos antiguos.** Son de 1999 a 2008 y de hospitales de Estados Unidos. Las prácticas clínicas, la codificación y las poblaciones han cambiado desde entonces.
- **Variables limitadas.** No incluye información posterior al alta ni determinantes sociales. El capítulo 13 muestra que el techo de desempeño viene de la información disponible, no del modelo.
- **Más lento en inferencia.** Tarda unos 0.45 ms por paciente, frente a unos 0.03 ms de un modelo individual, y necesita los cuatro modelos base y su conjunto de selección (12.9 MB).

## Riesgos y equidad

- **Atributos sensibles.** `race` y `gender` son predictores. El análisis de equidad del modelo base (capítulo 7) encontró, con una capacidad del 10 %, una cobertura de los readmitidos parecida por sexo y entre los grupos raciales más numerosos (relativa de 0.93 a 1.04). En cambio, la cobertura fue mucho menor:
  - en los encuentros sin raza registrada (0.36) y en la categoría «Other» (0.69);
  - en los pacientes de 10 a 20 años (0.27).

  La sección 16.7 repite ese análisis con DES-CPL y con XGBoost en el conjunto de prueba, con la misma capacidad (`results/equidad_des_cpl_prueba.csv`). Antes de cualquier uso real habría que monitorear la cobertura por grupo y decidir si excluir los atributos sensibles.
- **Daño por error.** Un falso negativo deja sin seguimiento a un paciente que se readmite. Un falso positivo consume capacidad del equipo. El umbral debe ajustarse a la capacidad real del servicio, no fijarse en 0.5.

## Usos no recomendados

- Negar, retrasar o condicionar atención a un paciente según su puntaje.
- Usarlo como diagnóstico o como sustituto del juicio clínico.
- Aplicarlo a poblaciones distintas de la de entrenamiento (otros países, otras enfermedades principales, pacientes pediátricos) sin volver a validarlo.
- Fijar primas, coberturas o decisiones administrativas sobre personas individuales.
