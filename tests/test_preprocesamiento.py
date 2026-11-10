"""Pruebas del preprocesador (src/preprocesamiento.py)."""

import numpy as np

from src.preprocesamiento import construir_preprocesador


def test_aprende_solo_de_los_datos_con_que_se_ajusta(datos, roles):
    """La mediana del escalado es la del subconjunto de ajuste, no la global.

    Es la propiedad que impide la fuga por preprocesamiento: dentro de un
    fold, el preprocesador solo puede conocer los datos de ese fold.
    """
    X, _, _ = datos
    subconjunto = X.iloc[:200]
    preprocesador = construir_preprocesador(roles, salida_densa=True)
    preprocesador.fit(subconjunto)

    columnas = roles["numericas"] + roles["ordinales"]
    escalado = preprocesador.named_transformers_["num"].named_steps["escalado"]
    np.testing.assert_allclose(escalado.center_,
                               subconjunto[columnas].median().to_numpy())


def test_categoria_nueva_al_predecir_no_falla(datos, roles):
    X, _, _ = datos
    preprocesador = construir_preprocesador(roles, salida_densa=True)
    columnas = preprocesador.fit_transform(X).shape[1]

    nuevo = X.iloc[:5].copy()
    nuevo["cat_a"] = "NIVEL_NUEVO"
    nuevo["race"] = "Raza_nueva"
    salida = preprocesador.transform(nuevo)
    assert salida.shape == (5, columnas)


def test_imputa_faltantes_numericos(datos, roles):
    X, _, _ = datos
    con_faltantes = X.copy()
    con_faltantes.loc[con_faltantes.index[:10], "num_a"] = np.nan
    salida = construir_preprocesador(roles, salida_densa=True) \
        .fit_transform(con_faltantes)
    assert not np.isnan(salida).any()


def test_race_no_se_agrupa_y_las_demas_si(datos, roles):
    """race conserva sus niveles raros; las demás categóricas los agrupan.

    Es la decisión del capítulo 2: race queda desagregada para el análisis
    de equidad, aunque un nivel tenga menos del 1 % de frecuencia.
    """
    X, _, _ = datos
    X = X.copy()
    X.loc[X.index[:2], "race"] = "Rara"
    X.loc[X.index[:2], "cat_a"] = "RARA"
    preprocesador = construir_preprocesador(roles, salida_densa=True).fit(X)

    niveles_race = preprocesador.named_transformers_["race"].categories_[0]
    assert "Rara" in list(niveles_race)

    codificador = (preprocesador.named_transformers_["cat"]
                   .named_steps["codificacion"])
    poco_frecuentes = codificador.infrequent_categories_[0]  # cat_a
    assert poco_frecuentes is not None and "RARA" in list(poco_frecuentes)


def test_salida_densa_es_un_arreglo(datos, roles):
    X, _, _ = datos
    salida = construir_preprocesador(roles, salida_densa=True).fit_transform(X)
    assert isinstance(salida, np.ndarray)
