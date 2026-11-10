# Protocolo experimental pre-registrado: modelo original DES-CPL

**Proyecto:** Readmisión temprana en pacientes diabéticos (Diabetes 130-US Hospitals)
**Entregable:** 3, bloques C y D
**Versión:** 1.0 — BORRADOR. Se congela con la etiqueta de Git `protocolo-v1` antes de ejecutar los experimentos definitivos.
**Fecha de registro:** _(la de la etiqueta de Git)_

Este documento fija, antes de ver los resultados definitivos, qué se va a probar, con qué datos, con qué pruebas y con qué presupuesto (guía del Entregable 3, sección 6.1). Cualquier desviación posterior se declara y se justifica en el capítulo 13; los análisis no previstos aquí se reportan como exploratorios y no sustentan por sí solos la condición R3.

## 0. Qué ocurrió antes de este registro (declaración)

El diseño del modelo se iteró de forma exploratoria, con los hiperparámetros de los modelos base del Entregable 2 (8-12 evaluaciones) y sobre las mismas particiones externas:

1. Formulación inicial de DES-CPL: piloto en el fold externo 0 y luego en los cinco folds (`results/des_cpl_evaluacion_v1.csv`). No superó al mejor modelo base.
2. Diagnóstico de errores (`results/des_cpl_diagnostico_errores.csv`), cribado de familias de modelos (`results/cribado_modelos_*.csv`) y estudio de información adicional (`results/limite_informativo.csv`).
3. Versión con poda y encogimiento elegidos por rejilla. Superó a XGBoost en la media sin significancia.

Estos resultados son exploratorios y se reportan completos en el capítulo 13. El conjunto de prueba reservado (20 % de los pacientes) **no se ha usado** para ninguna decisión de diseño.

## 1. Hipótesis

**Confirmatoria (única que sustenta R3).**
H1: el AUC-PR de DES-CPL es mayor que el del mejor benchmark.
H0: el AUC-PR de DES-CPL no es mayor que el del mejor benchmark.

El mejor benchmark es el de mayor AUC-PR media del bucle externo entre los nueve benchmarks: los siete modelos base con presupuesto de 60 evaluaciones (capítulo 12) y los dos del estado del arte (sección 4). Entre los siete modelos base, con los resultados ya registrados, es XGBoost (ADASYN), con 0,2290. Los del estado del arte se ejecutan después de congelar este protocolo.

**Exploratorias.**
- Aporte de cada componente: ablación de encogimiento, poda, ponderación por prevalencia, competencia por Brier, estratificación y exclusión del paciente.
- DES-CPL frente a los ensambles estáticos: promedio simple, pesos globales y apilamiento.
- Valor de λ elegido en cada fold, como medida de heterogeneidad local.

## 2. Métricas

- **Principal:** AUC-PR (precisión media), que mide el ordenamiento de la clase minoritaria (11,4 % de positivos).
- **Secundarias:** AUC-ROC, puntuación de Brier, error de calibración esperado (ECE, 10 bins) y, con el umbral operativo del capítulo 10 (percentil 85, que marca al 15 % de mayor riesgo), sensibilidad y precisión.

## 3. Datos y particiones

- **Problema principal.** Partición entrenamiento/prueba por paciente del capítulo 2 (80/20, semilla 42). En el entrenamiento, las cinco particiones externas `StratifiedGroupKFold(5, shuffle=True, random_state=42)` agrupadas por `patient_nbr`, las mismas del Entregable 2.
- **Conjunto de selección de DES-CPL.** Dentro de cada fold externo, predicciones fuera de pliegue de los modelos base con `StratifiedGroupKFold(3, shuffle=True, random_state=42)`.
- **Repeticiones.** Una repetición del bucle externo de cinco pliegues. La guía recomienda 5×5; no se adopta porque repetir los siete modelos base con validación anidada 25 veces excede el cómputo disponible antes de la entrega. Se declara como limitación de potencia.
- **Conjunto de prueba.** Se usa una sola vez, al final, para la prueba de DeLong y los intervalos bootstrap. Los modelos se reentrenan con todo el entrenamiento.
- **Conjuntos adicionales.** Se eligieron sin conocer el desempeño del modelo en ellos, con estos criterios: clasificación binaria desbalanceada, fuente pública (UCI), dominio y tamaño distintos.
  1. CDC Diabetes Health Indicators: 253.680 registros, 13,9 % de positivos, salud poblacional.
  2. Bank Marketing (Moro et al., 2014): 45.211 registros, 11,7 %, mercadeo.
  3. Default of Credit Card Clients (Yeh y Lien, 2009): 30.000 registros, 22,1 %, finanzas.

  Ninguno tiene identificador de individuo, así que su partición es estratificada simple, con 5 folds externos y semilla 42.

## 4. Modelos y presupuesto de ajuste

- **Benchmarks.** Los siete modelos base del Entregable 2 con su mejor combinación de balanceo. En cada fold, sus hiperparámetros son los que eligió Random Search con 60 evaluaciones en el estudio de convergencia del capítulo 12. Se suman dos métodos recientes del estado del arte para clasificación tabular, ambos no neuronales:
  - **LightGBM** (Ke et al., 2017);
  - **CatBoost** (Prokhorenkova et al., 2018).

  Se eligieron porque la literatura comparativa los señala como la referencia en datos tabulares de tamaño medio (Grinsztajn et al., 2022; Shwartz-Ziv y Armon, 2022). Reciben el mismo protocolo que los modelos base en el capítulo 12:
  - mismo preprocesador y mismas particiones;
  - Random Search con 60 evaluaciones sobre un cuarto de los pacientes, como XGBoost y Random Forest;
  - reajuste con todo el fold;
  - el tratamiento del desbalance como hiperparámetro de la búsqueda (pesos por clase en LightGBM; peso de la clase positiva entre 1 y 10 en CatBoost).

  Espacios en `src/sota.py`; ejecución en `experiments/benchmarks_sota.py`.
- **DES-CPL.** Combina los modelos base. Busca sus hiperparámetros con optimización bayesiana (Optuna, TPE, semilla 42 + número de fold) con **60 evaluaciones** por fold, maximizando el AUC-PR en el conjunto de selección. El espacio es:
  - k ∈ {5, 10, 20, 30, 50, 100};
  - τ ∈ {0, 10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³, 10⁻², 3·10⁻², 0,1, 0,3, ∞};
  - λ ∈ {0; 0,25; 0,5; 0,75; 0,9; 1};
  - poda a 4 o a 7 modelos.
- **Ablaciones.** Cada una con su propia búsqueda de 60 evaluaciones sobre el espacio restante.
- **Reporte del presupuesto real (guía, sección 6.2).** El informe incluye una tabla con las evaluaciones de hiperparámetros consumidas, el tiempo de entrenamiento y el tiempo de inferencia de cada modelo, y el hardware (4 núcleos físicos, `N_JOBS = 4`).
  - Para DES-CPL se separan sus 60 evaluaciones propias del costo de los modelos base que combina: sus búsquedas de 60 evaluaciones cada una y los 3 reentrenamientos por fold que producen el conjunto de selección.
  - La comparación es equitativa en evaluaciones propias (60 frente a 60). Que el costo total del ensamble sea mayor que el de un benchmark individual es inherente a cualquier ensamble y se declara, no se oculta.

## 5. Pruebas de hipótesis

Nivel de significancia α = 0,05. Para cada prueba se reportan el p-valor crudo y el ajustado, y el número total de comparaciones.

| # | Prueba | Datos | Comparación | Tipo |
|---|---|---|---|---|
| 1 | Friedman con Nemenyi y diagrama de diferencia crítica | AUC-PR de los 5 folds externos | Todos los modelos, DES-CPL incluido | Global |
| 2 | t corregida de Nadeau-Bengio, unilateral (razón prueba/entrenamiento 1/4) | Diferencias pareadas de AUC-PR por fold | DES-CPL frente a cada benchmark | Pareada |
| 3 | DeLong, unilateral | AUC-ROC en el conjunto de prueba | DES-CPL frente al mejor benchmark | Conjunto de prueba |

- **Corrección por comparaciones múltiples:** Holm-Bonferroni sobre las comparaciones de DES-CPL frente a los benchmarks. Las de la ablación se reportan con su propia corrección.
- **Tamaño del efecto:** δ de Cliff y g de Hedges sobre los folds. Despreciable si |δ| < 0,147.
- **Intervalo de confianza:** bootstrap BCa de la diferencia de AUC-PR en el conjunto de prueba, remuestreando pacientes, con 2.000 réplicas.
- **Varios conjuntos:** Friedman con corrección de Iman-Davenport y Wilcoxon-Holm por pares sobre los cuatro conjuntos.

**Superioridad (guía, sección 6.4).** Se declara solo si se cumplen a la vez cuatro condiciones:
1. AUC-PR media mayor que la del mejor benchmark y que la de cada uno de los siete modelos base.
2. Las tres pruebas rechazan H0 con p ajustado < 0,05.
3. |δ de Cliff| ≥ 0,147.
4. DES-CPL es mejor en al menos dos de los tres conjuntos adicionales y no es significativamente peor en ninguno.

## 6. Robustez y calidad probabilística

- **Sensibilidad.** AUC-PR al variar k, τ y λ alrededor de los valores elegidos, en el fold 0.
- **Semillas.** Diez semillas de la búsqueda de Optuna y de la partición interna del conjunto de selección para DES-CPL y para el mejor benchmark: 42, 7, 2026, 1, 2, 3, 4, 5, 6 y 8.
- **Ruido, desbalance y tamaño.** Inversión del 5 % y del 10 % de las etiquetas de entrenamiento, submuestreo de positivos hasta un 5 % de prevalencia y entrenamiento con el 25 % y el 50 % de los pacientes.
- **Calibración.** Curvas de fiabilidad, puntuación de Brier y ECE.
- **Predicción conforme.** Cobertura empírica de conjuntos de predicción con α = 0,1, con conformal inductivo sobre un conjunto de calibración separado del entrenamiento.

## 7. Software y reproducibilidad

- **Entorno:** `requirements.txt` con versiones exactas (Python 3.12.13).
- **Semilla principal:** 42.
- **Código:** `src/des_cpl.py` (modelo) y `src/experimento_des_cpl.py` (protocolo), con pruebas en `tests/test_des_cpl.py` y `tests/test_experimento_des_cpl.py`.
- **Comandos definitivos del problema principal:** `python experiments/benchmarks_sota.py` y `python experiments/des_cpl.py --folds 0 1 2 3 4`.
