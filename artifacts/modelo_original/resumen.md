### DES-CPL: selección dinámica de ensambles por competencia probabilística local

**Idea.** Los mejores modelos del benchmark empatan en promedio. DES-CPL decide, para cada paciente, cuánto confiar en cada modelo según la calidad de sus probabilidades (puntuación de Brier) en pacientes parecidos, y encoge esa competencia local hacia la global cuando la local es ruidosa.

**Algoritmo** (capítulo 15):
1. Predicciones fuera de pliegue de los modelos base, en 3 pliegues agrupados por paciente.
2. Poda a los modelos de mayor AUC-PR fuera de pliegue: arbol, random_forest, xgboost, svm.
3. Región de competencia: los $k$ vecinos positivos y los $k$ negativos en el espacio de logits de los modelos, sin el propio paciente.
4. Competencia: el Brier local, reponderado por la prevalencia local estimada.
5. Encogimiento hacia la competencia global: $\hat C = (1-\lambda)\hat C^{loc} + \lambda C^{glob}$.
6. Pesos $w_m \propto \exp(-\hat C_m/\tau)$ y predicción $\hat p = \sum_m w_m p_m$.

**Configuración del artefacto:** $k = 20$, $\tau = 0.0003$, $\lambda = 0.9$, elegidos con 60 evaluaciones de Optuna.

**Desempeño:**
- Bucle externo, 5 folds: AUC-PR 0.2279 ± 0.0174.
- Prueba: AUC-PR 0.2293, AUC-ROC 0.6661.
- Supera de forma significativa a 6 de los 7 modelos base en prueba (DeLong con Holm) y empata con XGBoost, LightGBM y CatBoost.

**Evaluación ampliada** (capítulo 16):
- Es 3.º en rango medio entre 13 métodos y 4 conjuntos de datos, por delante de todos los benchmarks individuales.
- Pierde 1.6 a 1.9 veces menos que XGBoost ante ruido en las etiquetas, menos prevalencia y menos datos.
- En 10 semillas su AUC-PR es 0.2270 ± 0.0007, frente a 0.2251 ± 0.0017 de XGBoost.
