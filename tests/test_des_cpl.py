"""Pruebas del modelo original DES-CPL (src/des_cpl.py)."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.des_cpl import (DESCPL, RegionCompetencia, combinar,
                         competencia_global, competencia_local,
                         competencias_locales, perfil, pesos_softmax, podar,
                         predecir_con_seleccion)


# --- Componentes ------------------------------------------------------------

def test_perfil_finito_con_ceros_y_unos():
    """El recorte evita infinitos con los ceros y unos exactos del árbol."""
    phi = perfil([[0.0, 1.0, 0.5]])
    assert np.isfinite(phi).all()
    assert phi[0, 2] == 0.0
    assert phi[0, 0] == pytest.approx(-phi[0, 1])


def test_competencia_brier_calculada_a_mano():
    """Brier balanceado: 1/2 promedio en positivos + 1/2 en negativos."""
    # Una consulta, 2 positivos y 2 negativos, dos modelos.
    p = np.array([[[0.9, 0.5], [0.7, 0.5], [0.1, 0.5], [0.3, 0.5]]])
    y = np.array([[1, 1, 0, 0]])
    c = competencia_local(p, y, "brier")
    esperado_m0 = 0.5 * ((0.1**2 + 0.3**2) / 2) + 0.5 * ((0.1**2 + 0.3**2) / 2)
    assert c[0, 0] == pytest.approx(esperado_m0)
    assert c[0, 1] == pytest.approx(0.25)


def test_brier_distingue_lo_que_la_exactitud_no():
    """Dos modelos que fallan la etiqueta: Brier premia al que se acerca más."""
    p = np.array([[[0.45, 0.05], [0.2, 0.2]]])
    y = np.array([[1, 0]])
    brier = competencia_local(p, y, "brier")
    exactitud = competencia_local(p, y, "exactitud")
    assert brier[0, 0] < brier[0, 1]
    assert exactitud[0, 0] == exactitud[0, 1]


def test_competencia_balanceada_no_depende_del_numero_por_clase():
    """Con pérdidas iguales dentro de cada clase, el número de vecinos no importa."""
    p1 = np.array([[[0.8], [0.2]]])
    p2 = np.array([[[0.8], [0.8], [0.8], [0.2]]])
    c1 = competencia_local(p1, np.array([[1, 0]]))
    c2 = competencia_local(p2, np.array([[1, 1, 1, 0]]))
    assert c1[0, 0] == pytest.approx(c2[0, 0])


def test_peso_de_positivos_por_consulta():
    """Con q por consulta, la competencia es q*pérdida+ + (1-q)*pérdida-."""
    p = np.array([[[0.8], [0.2]], [[0.8], [0.2]]])
    y = np.array([[1, 0], [1, 0]])
    c = competencia_local(p, y, "brier", np.array([0.1, 0.9]))
    np.testing.assert_allclose(c[:, 0], [0.1 * 0.04 + 0.9 * 0.04] * 2)
    p = np.array([[[0.9], [0.5]]])
    c = competencia_local(p, np.array([[1, 0]]), "brier", np.array([0.2]))
    assert c[0, 0] == pytest.approx(0.2 * 0.01 + 0.8 * 0.25)


def test_brier_balanceado_premia_la_probabilidad_cercana_a_05():
    """Documenta por qué la ponderación por defecto es la prevalencia local.

    En una zona de bajo riesgo, un modelo calibrado (0.05) y uno inflado
    (0.40): con peso 1/2 por clase gana el inflado; con la prevalencia
    local de la zona (5 %), gana el calibrado.
    """
    k = 20
    p_pos = np.tile([[0.05, 0.40]], (k, 1))
    p_neg = np.tile([[0.05, 0.40]], (k, 1))
    p = np.concatenate([p_neg, p_pos])[None]
    y = np.r_[np.zeros(k), np.ones(k)][None]
    balanceada = competencia_local(p, y, "brier", 0.5)
    prevalencia = competencia_local(p, y, "brier", np.array([0.05]))
    assert balanceada[0, 1] < balanceada[0, 0]
    assert prevalencia[0, 0] < prevalencia[0, 1]


def test_prevalencia_local_suavizada(seleccion):
    perfiles, y, grupos = seleccion
    region = RegionCompetencia(perfiles, y, grupos)
    q = region.prevalencia_local(perfiles[:50], k=10, grupos_consulta=grupos[:50])
    assert ((q > 0) & (q < 1)).all()
    q_infinito = region.prevalencia_local(perfiles[:5], k=10, suavizado=1e9)
    np.testing.assert_allclose(q_infinito, y.mean(), rtol=1e-6)


def test_ponderacion_desconocida(seleccion):
    perfiles, y, grupos = seleccion
    region = RegionCompetencia(perfiles, y, grupos)
    prob = np.full((len(y), 2), 0.3)
    with pytest.raises(ValueError):
        predecir_con_seleccion(region, prob, prob[:3], perfiles[:3], 5, 0.01,
                               ponderacion="otra")


def test_competencia_desconocida():
    with pytest.raises(ValueError):
        competencia_local(np.zeros((1, 2, 1)), np.zeros((1, 2)), "auc")


def test_temperatura_infinita_es_el_promedio_simple():
    c = np.array([[0.10, 0.20, 0.30]])
    probs = np.array([[0.2, 0.4, 0.9]])
    w = pesos_softmax(c, np.inf)
    assert combinar(w, probs)[0] == pytest.approx(probs.mean())


def test_temperatura_cero_selecciona_el_mejor():
    c = np.array([[0.30, 0.10, 0.20], [0.1, 0.1, 0.5]])
    w = pesos_softmax(c, 0)
    np.testing.assert_allclose(w, [[0, 1, 0], [0.5, 0.5, 0]])


def test_pesos_son_una_combinacion_convexa():
    aleatorio = np.random.default_rng(0)
    c = aleatorio.uniform(0, 0.3, size=(50, 4))
    probs = aleatorio.uniform(size=(50, 4))
    for tau in (1e-4, 0.01, 1.0):
        w = pesos_softmax(c, tau)
        assert (w >= 0).all()
        np.testing.assert_allclose(w.sum(axis=1), 1)
        p = combinar(w, probs)
        assert (p >= probs.min(axis=1) - 1e-12).all()
        assert (p <= probs.max(axis=1) + 1e-12).all()


def test_softmax_estable_con_temperatura_minima():
    w = pesos_softmax(np.array([[0.0, 1000.0]]), 1e-8)
    assert np.isfinite(w).all()
    np.testing.assert_allclose(w, [[1, 0]])


def test_descomposicion_de_ambiguedad():
    """Proposición 1: pérdida del ensamble = media ponderada - dispersión."""
    aleatorio = np.random.default_rng(1)
    probs = aleatorio.uniform(size=(200, 5))
    w = pesos_softmax(aleatorio.uniform(0, 0.3, size=(200, 5)), 0.05)
    y = aleatorio.integers(0, 2, size=200)
    p = combinar(w, probs)
    izquierda = (p - y) ** 2
    derecha = ((w * (probs - y[:, None]) ** 2).sum(axis=1)
               - (w * (probs - p[:, None]) ** 2).sum(axis=1))
    np.testing.assert_allclose(izquierda, derecha, atol=1e-12)
    assert (izquierda <= (w * (probs - y[:, None]) ** 2).sum(axis=1) + 1e-12).all()


# --- Región de competencia --------------------------------------------------

@pytest.fixture
def seleccion():
    aleatorio = np.random.default_rng(2)
    n = 400
    grupos = aleatorio.integers(0, 120, size=n)
    y = (aleatorio.uniform(size=n) < 0.15).astype(int)
    perfiles = aleatorio.normal(size=(n, 3))
    return perfiles, y, grupos


def test_region_estratificada_tiene_k_de_cada_clase(seleccion):
    perfiles, y, grupos = seleccion
    region = RegionCompetencia(perfiles, y, grupos)
    idx = region.region(perfiles[:30], k=7)
    assert idx.shape == (30, 14)
    assert (y[idx[:, :7]] == 0).all() and (y[idx[:, 7:]] == 1).all()


def test_region_excluye_al_propio_paciente(seleccion):
    perfiles, y, grupos = seleccion
    region = RegionCompetencia(perfiles, y, grupos)
    idx = region.region(perfiles, k=5, grupos_consulta=grupos)
    assert (grupos[idx] != grupos[:, None]).all()
    assert idx.shape == (len(y), 10)
    todos = region.vecinos(perfiles, k=10, grupos_consulta=grupos)
    assert (grupos[todos] != grupos[:, None]).all()


def test_region_sin_estratificar(seleccion):
    perfiles, y, grupos = seleccion
    region = RegionCompetencia(perfiles, y, grupos)
    idx = region.region(perfiles[:10], k=4, estratificado=False)
    assert idx.shape == (10, 8)


def test_region_respeta_la_cercania(seleccion):
    """Sin exclusión, el vecino más cercano de un punto del conjunto es él mismo."""
    perfiles, y, grupos = seleccion
    region = RegionCompetencia(perfiles, y, grupos)
    idx = region.vecinos(perfiles[:20], k=3)
    np.testing.assert_array_equal(idx[:, 0], np.arange(20))


def test_competencia_premia_al_modelo_bueno_localmente():
    """Modelo 0 acierta en la mitad izquierda del perfil y 1 en la derecha."""
    aleatorio = np.random.default_rng(3)
    n = 600
    lado = aleatorio.uniform(-1, 1, size=n)
    y = (aleatorio.uniform(size=n) < 0.3).astype(int)
    bueno = np.where(y == 1, 0.9, 0.1)
    malo = np.full(n, 0.5)
    prob = np.column_stack([np.where(lado < 0, bueno, malo),
                            np.where(lado < 0, malo, bueno)])
    perfiles = np.column_stack([lado, np.zeros(n)])
    region = RegionCompetencia(perfiles, y, np.arange(n))
    _, det = predecir_con_seleccion(region, prob, prob, perfiles, k=10,
                                    tau=0.01, grupos_consulta=np.arange(n),
                                    detalle=True)
    izquierda = lado < -0.2
    derecha = lado > 0.2
    assert det["pesos"][izquierda, 0].mean() > 0.9
    assert det["pesos"][derecha, 1].mean() > 0.9


# --- Estimador ----------------------------------------------------------------

@pytest.fixture
def datos_tabulares():
    aleatorio = np.random.default_rng(4)
    n_pacientes = 250
    grupos = np.repeat(np.arange(n_pacientes), aleatorio.integers(1, 4, n_pacientes))
    n = len(grupos)
    X = aleatorio.normal(size=(n, 4))
    logit = -2 + 1.5 * X[:, 0] - X[:, 1]
    y = (aleatorio.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y, grupos


def _ensamble(**kw):
    return DESCPL([("logistica", LogisticRegression()),
                   ("constante", DummyClassifier(strategy="prior"))], **kw)


def test_ajuste_y_probabilidades_validas(datos_tabulares):
    X, y, grupos = datos_tabulares
    modelo = _ensamble(k=10, tau=0.01).fit(X, y, grupos)
    proba = modelo.predict_proba(X[:50])
    assert proba.shape == (50, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert not np.isnan(modelo.prob_seleccion_).any()
    assert len(modelo.modelos_pliegue_) == 3


def test_favorece_al_modelo_informativo(datos_tabulares):
    X, y, grupos = datos_tabulares
    modelo = _ensamble(k=10, tau=0.01).fit(X, y, grupos)
    _, det = modelo.predict_proba(X, detalle=True)
    assert det["pesos"][:, 0].mean() > 0.6


def test_misma_semilla_mismo_resultado(datos_tabulares):
    X, y, grupos = datos_tabulares
    a = _ensamble(k=8).fit(X, y, grupos).predict_proba(X)
    b = _ensamble(k=8).fit(X, y, grupos).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_seleccion_fuera_de_pliegue_agrupada(datos_tabulares):
    """Ningún paciente se predice con un modelo de pliegue que lo vio."""
    X, y, grupos = datos_tabulares
    modelo = _ensamble()
    for entrenamiento, validacion in modelo._particiones(X, y, grupos):
        assert not set(grupos[entrenamiento]) & set(grupos[validacion])


def test_puntuar_seleccion_y_extremos_de_temperatura(datos_tabulares):
    X, y, grupos = datos_tabulares
    modelo = _ensamble(k=10).fit(X, y, grupos)
    from sklearn.metrics import average_precision_score
    promedio = average_precision_score(y, modelo.prob_seleccion_.mean(axis=1))
    assert modelo.puntuar_seleccion(tau=np.inf) == pytest.approx(promedio)
    assert 0 < modelo.puntuar_seleccion(tau=0.01) <= 1
    sin_exclusion = clone(modelo).set_params(excluir_paciente=False)
    sin_exclusion.fit(X, y, grupos)
    assert 0 < sin_exclusion.puntuar_seleccion() <= 1


def test_compatible_con_clone(datos_tabulares):
    modelo = _ensamble(k=12, tau=0.05)
    copia = clone(modelo)
    assert copia.get_params()["k"] == 12
    assert copia.get_params()["tau"] == 0.05


def test_vecindario_recortado_igual_al_de_k_menor(seleccion):
    """La búsqueda consulta una vez con k máximo y recorta: debe dar lo mismo."""
    perfiles, y, grupos = seleccion
    region = RegionCompetencia(perfiles, y, grupos)
    grande = region.vecindario(perfiles, 15, grupos)
    chico = region.vecindario(perfiles, 6, grupos)
    np.testing.assert_array_equal(grande[0][:, :6], chico[0])
    np.testing.assert_array_equal(grande[1][:, :6], chico[1])
    np.testing.assert_array_equal(grande[None][:, :12], chico[None])



# --- Encogimiento y poda ----------------------------------------------------------

def test_competencia_global_es_el_brier_de_cada_modelo():
    p = np.array([[0.9, 0.5], [0.2, 0.5], [0.4, 0.5]])
    y = np.array([1, 0, 0])
    esperado = ((p - y[:, None]) ** 2).mean(axis=0)
    np.testing.assert_allclose(competencia_global(p, y)[0], esperado)


def test_encogimiento_total_da_los_mismos_pesos_a_todos(seleccion):
    """Con λ = 1 la competencia ya no depende de la consulta: pesos globales."""
    perfiles, y, grupos = seleccion
    aleatorio = np.random.default_rng(5)
    prob = aleatorio.uniform(size=(len(y), 3))
    region = RegionCompetencia(perfiles, y, grupos)
    vec = region.vecindario(perfiles[:40], 8)
    c = competencias_locales(region, vec, prob, 8, encogimiento=1.0)[0]
    np.testing.assert_allclose(c, np.repeat(competencia_global(prob, y), 40, axis=0))
    c0 = competencias_locales(region, vec, prob, 8, encogimiento=0.0)[0]
    c5 = competencias_locales(region, vec, prob, 8, encogimiento=0.5)[0]
    np.testing.assert_allclose(c5, 0.5 * c0 + 0.5 * competencia_global(prob, y))
    with pytest.raises(ValueError):
        competencias_locales(region, vec, prob, 8, encogimiento=1.5)


def test_poda_conserva_los_mejores_en_orden():
    aleatorio = np.random.default_rng(6)
    y = (aleatorio.uniform(size=500) < 0.3).astype(int)
    ruido = aleatorio.uniform(size=(500, 3))
    bueno = np.clip(0.3 + 0.4 * (y - 0.3) + 0.1 * aleatorio.normal(size=500), 0, 1)
    prob = np.column_stack([ruido[:, 0], bueno, ruido[:, 1], ruido[:, 2]])
    activos = podar(prob, y, 2)
    assert 1 in activos and len(activos) == 2
    assert list(activos) == sorted(activos)
    np.testing.assert_array_equal(podar(prob, y, None), np.arange(4))


def test_estimador_con_poda_y_encogimiento(datos_tabulares):
    X, y, grupos = datos_tabulares
    modelo = DESCPL([("logistica", LogisticRegression()),
                     ("constante", DummyClassifier(strategy="prior")),
                     ("logistica_debil", LogisticRegression(C=1e-4))],
                    k=8, tau=0.01, encogimiento=0.5, n_modelos=2).fit(X, y, grupos)
    assert len(modelo.activos_) == 2
    proba = modelo.predict_proba(X[:30])
    assert proba.shape == (30, 2)
    assert 0 < modelo.puntuar_seleccion(encogimiento=1.0) <= 1


# --- Entradas inválidas (requisito iii de la guía) -------------------------------

@pytest.mark.parametrize("parametros", [
    {"k": 0}, {"k": 2.5}, {"tau": -0.1}, {"encogimiento": 1.2},
    {"n_modelos": 5}, {"competencia": "auc"}, {"ponderacion": "otra"},
])
def test_rechaza_hiperparametros_invalidos(datos_tabulares, parametros):
    X, y, grupos = datos_tabulares
    with pytest.raises(ValueError):
        _ensamble(**parametros).fit(X, y, grupos)


def test_rechaza_etiquetas_y_tamanos_invalidos(datos_tabulares):
    X, y, grupos = datos_tabulares
    with pytest.raises(ValueError):
        _ensamble().fit(X, np.where(y == 1, 2, 0), grupos)   # no binaria
    with pytest.raises(ValueError):
        _ensamble().fit(X, np.zeros_like(y), grupos)         # una sola clase
    with pytest.raises(ValueError):
        _ensamble().fit(X[:-5], y, grupos)                   # tamaños distintos
    with pytest.raises(ValueError):
        _ensamble().fit(X, y, grupos[:-5])
    with pytest.raises(ValueError):
        DESCPL([]).fit(X, y, grupos)


def test_predecir_sin_ajustar_falla(datos_tabulares):
    from sklearn.exceptions import NotFittedError
    X, _, _ = datos_tabulares
    with pytest.raises(NotFittedError):
        _ensamble().predict_proba(X)
