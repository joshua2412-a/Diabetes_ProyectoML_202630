"""
Configuración del dashboard: rutas, enlaces, autoría y nombres legibles.

Todo lo que el tablero escribe sobre el proyecto sale de aquí o de los
archivos de datos y artefactos; no hay rutas ni nombres repartidos por las
páginas.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parents[2]
PROCESADOS = RAIZ / "data" / "processed"
ARTEFACTOS = RAIZ / "artifacts"
RESULTADOS = RAIZ / "results"

TRAIN = PROCESADOS / "diabetes_train.parquet"
PRUEBA = PROCESADOS / "diabetes_test.parquet"
ROLES = PROCESADOS / "roles_variables.json"
TABLA_MAESTRA = RESULTADOS / "tabla_maestra_clasificacion.csv"
MODEL_CARD = RAIZ / "MODEL_CARD.md"
ORIGINAL = ARTEFACTOS / "modelo_original"

# --------------------------------------------------------------------------- #
# Enlaces y autoría
# --------------------------------------------------------------------------- #
URL_LIBRO = "https://joshua2412-a.github.io/Diabetes_ProyectoML_202630/"
URL_REPO = "https://github.com/joshua2412-a/Diabetes_ProyectoML_202630"
URL_DATOS = ("https://archive.ics.uci.edu/dataset/296/"
             "diabetes+130-us+hospitals+for+years+1999-2008")
DOI_DATOS = "https://doi.org/10.24432/C5230J"
AUTORES = ["Joshua Hincapié Llorente", "Alejandro Cantillo Escorcia"]
INSTITUCION = "Universidad del Norte · Machine Learning"
VERSION = "1.1.0"
FECHA_ACTUALIZACION = "noviembre de 2026"

# --------------------------------------------------------------------------- #
# Clases y modelos
# --------------------------------------------------------------------------- #
CLASES = {0: "No readmitido", 1: "Readmitido en < 30 días"}
CLASES_CORTAS = {0: "No readmitido", 1: "Readmitido"}

MODELOS = {
    "xgboost": "XGBoost", "random_forest": "Random Forest", "svm": "SVM lineal",
    "logistica": "Regresión logística", "knn": "k-NN", "arbol": "Árbol de decisión",
    "bayes": "Naive Bayes", "des_cpl": "DES-CPL (modelo original)",
    # Benchmarks del estado del arte: solo aparecen en la comparación del
    # protocolo (no tienen artefacto serializado en el dashboard).
    "lightgbm": "LightGBM", "catboost": "CatBoost",
}
#: Clave del modelo original del Entregable 3. Las vistas la usan para
#: destacarlo y para no mezclarlo con los benchmarks del Entregable 2, que se
#: buscaron con otro presupuesto (ver «Comparación estadística»).
MODELO_ORIGINAL = "des_cpl"
#: Nombres para frases y rótulos: sin «(modelo original)», que ya dice el
#: contexto y que dejaba paréntesis dobles («DES-CPL (modelo original) (2,80)»).
MODELOS_CORTOS = {**MODELOS, MODELO_ORIGINAL: "DES-CPL"}


def nombre_modelo(clave: str) -> str:
    """Nombre corto y legible de un modelo (o la clave tal cual)."""
    return MODELOS_CORTOS.get(clave, clave)
BALANCEOS = {"ninguno": "Sin balanceo", "smote": "SMOTE", "adasyn": "ADASYN",
             "class_weight": "Pesos de clase"}
OPTIMIZADORES = {"grid": "Grid Search", "random": "Random Search",
                 "optuna": "Bayesiano (Optuna)", "deap": "Genético"}

#: Explicación local en vivo: permutaciones por predicción. Con 38
#: variables y 20 filas de fondo son 4,680 filas por consulta.
PERMUTACIONES_EN_VIVO = 6

# --------------------------------------------------------------------------- #
# Diccionario de variables: (nombre en la interfaz, descripción)
# Las columnas del parquet conservan sus nombres originales.
# --------------------------------------------------------------------------- #
VARIABLES = {
    "time_in_hospital": ("Días de hospitalización", "Días entre el ingreso y el alta (1 a 14)."),
    "num_lab_procedures": ("Pruebas de laboratorio", "Número de pruebas de laboratorio durante el ingreso."),
    "num_procedures": ("Procedimientos", "Procedimientos distintos de las pruebas de laboratorio."),
    "num_medications": ("Medicamentos distintos", "Nombres genéricos distintos administrados en el ingreso."),
    "number_outpatient": ("Consultas externas previas", "Visitas ambulatorias en el año anterior al ingreso."),
    "number_emergency": ("Urgencias previas", "Visitas a urgencias en el año anterior al ingreso."),
    "number_inpatient": ("Hospitalizaciones previas", "Hospitalizaciones en el año anterior al ingreso."),
    "number_diagnoses": ("Diagnósticos registrados", "Número de diagnósticos registrados en el sistema."),
    "n_farmacos_activos": ("Antidiabéticos activos", "Fármacos antidiabéticos prescritos en el ingreso (derivada)."),
    "age_ord": ("Edad", "Grupo de edad en decenios, codificado de forma ordinal."),
    "hospitalizacion_previa": ("Hospitalización previa", "1 si hubo al menos una hospitalización el año anterior (derivada)."),
    "A1Cresult": ("HbA1c", "Resultado de la hemoglobina glicosilada; incluye «no medido»."),
    "max_glu_serum": ("Glucosa sérica máxima", "Resultado de la glucosa sérica; incluye «no medido»."),
    "admission_source_id": ("Origen de la admisión", "De dónde llegó el paciente (urgencias, remisión...)."),
    "admission_type_id": ("Tipo de admisión", "Urgente, electiva, de emergencia..."),
    "discharge_disposition_id": ("Destino al alta", "A dónde fue el paciente al salir (casa, otro centro...)."),
    "especialidad": ("Especialidad del médico", "Especialidad del médico que admite; incluye «desconocida»."),
    "diag_1_grupo": ("Diagnóstico principal", "Capítulo ICD-9 del diagnóstico principal."),
    "diag_2_grupo": ("Diagnóstico secundario", "Capítulo ICD-9 del diagnóstico secundario."),
    "diag_3_grupo": ("Diagnóstico adicional", "Capítulo ICD-9 del tercer diagnóstico."),
    "gender": ("Sexo", "Sexo registrado."),
    "race": ("Grupo racial", "Atributo protegido; los faltantes se conservan como categoría."),
    "change": ("Cambio de medicación", "Si hubo cambios en la medicación antidiabética."),
    "diabetesMed": ("Medicación antidiabética", "Si se prescribió algún antidiabético."),
    "insulin": ("Insulina", "Prescripción y cambio de dosis de insulina."),
    "metformin": ("Metformina", "Prescripción y cambio de dosis."),
    "glimepiride": ("Glimepirida", "Prescripción y cambio de dosis."),
    "glipizide": ("Glipizida", "Prescripción y cambio de dosis."),
    "glyburide": ("Gliburida", "Prescripción y cambio de dosis."),
    "glyburide-metformin": ("Gliburida-metformina", "Prescripción y cambio de dosis."),
    "pioglitazone": ("Pioglitazona", "Prescripción y cambio de dosis."),
    "rosiglitazone": ("Rosiglitazona", "Prescripción y cambio de dosis."),
    "repaglinide": ("Repaglinida", "Prescripción y cambio de dosis."),
    "nateglinide": ("Nateglinida", "Prescripción y cambio de dosis."),
    "chlorpropamide": ("Clorpropamida", "Prescripción y cambio de dosis."),
    "acarbose": ("Acarbosa", "Prescripción y cambio de dosis."),
    "miglitol": ("Miglitol", "Prescripción y cambio de dosis."),
    "tolazamide": ("Tolazamida", "Prescripción y cambio de dosis."),
    "readmitted_30": ("Readmisión en < 30 días", "Variable objetivo: 1 si reingresó en menos de 30 días."),
}

#: Agrupación del formulario de predicción (en el orden del relato clínico).
#: Las variables derivadas (hospitalización previa y antidiabéticos activos)
#: no se piden: inferencia.derivar las calcula con la regla del ETL.
SECCIONES_FORMULARIO = [
    ("Paciente", ["age_ord", "gender", "race"]),
    ("Ingreso", ["admission_type_id", "admission_source_id", "especialidad",
                 "time_in_hospital", "discharge_disposition_id"]),
    ("Historia del año anterior", ["number_inpatient", "number_emergency",
                                   "number_outpatient"]),
    ("Clínica del ingreso", ["diag_1_grupo", "diag_2_grupo", "diag_3_grupo",
                             "number_diagnoses", "num_lab_procedures",
                             "num_procedures", "num_medications", "A1Cresult",
                             "max_glu_serum"]),
    ("Tratamiento", ["diabetesMed", "change", "insulin"]),
    ("Otros antidiabéticos", ["metformin", "glimepiride", "glipizide", "glyburide",
                              "glyburide-metformin", "pioglitazone", "rosiglitazone",
                              "repaglinide", "nateglinide", "chlorpropamide",
                              "acarbose", "miglitol", "tolazamide"]),
]


# --------------------------------------------------------------------------- #
# Niveles legibles. Los datos conservan los códigos originales (en inglés y,
# en algunos casos, con un espacio inicial); la interfaz los traduce aquí.
# --------------------------------------------------------------------------- #
_ESTADOS_FARMACO = {"No": "No prescrito", "Steady": "Dosis estable", "Up": "Dosis sube",
                    "Down": "Dosis baja", "Cambio": "Dosis cambia"}

NIVELES = {
    "admission_type_id": {"Emergency": "Emergencia", "Elective": "Electiva", "Urgent": "Urgente",
                          "NULL": "Sin dato", "Not Available": "No disponible", "Otro": "Otro"},
    "admission_source_id": {"Emergency Room": "Sala de urgencias", "Physician Referral": "Remisión médica",
                            "NULL": "Sin dato", "Transfer from a hospital": "Traslado de otro hospital",
                            "Transfer from another health care facility": "Traslado de otro centro",
                            "Clinic Referral": "Remisión de clínica", "Otro": "Otro"},
    "discharge_disposition_id": {
        "Discharged to home": "Casa",
        "Discharged/transferred to SNF": "Centro de enfermería",
        "Discharged/transferred to home with home health service": "Casa con atención domiciliaria",
        "Discharged/transferred to another short term hospital": "Otro hospital",
        "Discharged/transferred to another rehab fac including rehab units of a hospital .":
            "Centro de rehabilitación",
        "Discharged/transferred to another type of inpatient care institution": "Otra institución",
        "NULL": "Sin dato", "Otro": "Otro"},
    "gender": {"Female": "Mujer", "Male": "Hombre"},
    "race": {"Caucasian": "Caucásico", "AfricanAmerican": "Afroamericano", "Hispanic": "Hispano",
             "Asian": "Asiático", "Other": "Otro", "No informada": "No informada"},
    "A1Cresult": {"No medida": "No medida", "Norm": "Normal", ">7": "> 7 %", ">8": "> 8 %"},
    "max_glu_serum": {"No medida": "No medida", "Norm": "Normal", ">200": "> 200 mg/dL",
                      ">300": "> 300 mg/dL"},
    "change": {"No": "Sin cambio", "Ch": "Con cambio"},
    "diabetesMed": {"Yes": "Sí", "No": "No"},
}


def nivel(variable: str, valor) -> str:
    """Nivel legible de una variable categórica (o el valor tal cual)."""
    texto = str(valor)
    tabla = NIVELES.get(variable)
    if tabla is None and variable in _FARMACOS:
        tabla = _ESTADOS_FARMACO
    if tabla is None:
        return texto.strip()
    return tabla.get(texto, tabla.get(texto.strip(), texto.strip()))


_FARMACOS = {"metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
             "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
             "tolazamide", "insulin", "glyburide-metformin"}
ESTADOS_FARMACO = _ESTADOS_FARMACO

#: Referencia del artículo que presentó el conjunto de datos; su análisis
#: central es justamente la medición de la HbA1c y la readmisión.
FUENTE_HBA1C = ("Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission "
                "Rates: Analysis of 70,000 Clinical Database Patient Records. BioMed Research "
                "International, 2014, 781670.")


def nombre(variable: str) -> str:
    """Nombre legible de una columna."""
    return VARIABLES.get(variable, (variable, ""))[0]


def descripcion(variable: str) -> str:
    return VARIABLES.get(variable, ("", ""))[1]
