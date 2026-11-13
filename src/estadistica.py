"""
Pruebas estadísticas y tamaños del efecto para comparar modelos.

Reúne las funciones que los capítulos 11 y 12 definían dentro de los
notebooks: la diferencia crítica de Nemenyi, la prueba de DeLong, la
corrección de Holm, los intervalos bootstrap BCa por paciente, la delta de
Cliff y la d de Cohen. Se trasladaron sin cambios de lógica; aquí quedan
cubiertas por pruebas automáticas (``tests/test_estadistica.py``) y
disponibles para los capítulos del Entregable 3.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .configuracion import SEMILLA


# ==========================================================================
# Comparación de varios modelos sobre los mismos folds
# ==========================================================================

def diferencia_critica(k, n, alfa=0.05):
    """Diferencia crítica de Nemenyi para k modelos y n bloques.

    Usa la aproximación habitual con el valor crítico del rango
    studentizado dividido por la raíz de dos.

    Returns
    -------
    float
    """
    q_alfa = stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2)
    return float(q_alfa * np.sqrt(k * (k + 1) / (6 * n)))


# ==========================================================================
# Comparación de dos AUC sobre las mismas observaciones
# ==========================================================================

def _estructura_delong(y_real, probabilidades):
    """Componentes de la varianza del AUC según DeLong.

    Returns
    -------
    tuple
        AUC y las dos series de pseudovalores (V10 sobre positivos y V01
        sobre negativos) que definen su varianza.
    """
    positivos = probabilidades[y_real == 1]
    negativos = probabilidades[y_real == 0]
    m, n = len(positivos), len(negativos)

    # Kernel de comparación con empates ponderados a la mitad.
    comparaciones = (positivos[:, None] > negativos[None, :]).astype(float)
    comparaciones += 0.5 * (positivos[:, None] == negativos[None, :])

    auc = comparaciones.mean()
    v10 = comparaciones.mean(axis=1)   # un valor por positivo
    v01 = comparaciones.mean(axis=0)   # un valor por negativo
    return auc, v10, v01, m, n


def prueba_delong(y_real, probabilidades_a, probabilidades_b):
    """Prueba de DeLong para dos AUC correlacionados.

    Contrasta H0: AUC_A = AUC_B sobre las mismas observaciones, usando la
    representación del AUC como estadístico U y su matriz de covarianzas
    estimada a partir de los pseudovalores.

    Returns
    -------
    dict
        AUC de cada modelo, su diferencia, el estadístico z y el p-valor
        bilateral.
    """
    auc_a, v10_a, v01_a, m, n = _estructura_delong(y_real, probabilidades_a)
    auc_b, v10_b, v01_b, _, _ = _estructura_delong(y_real, probabilidades_b)

    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    covarianza = s10 / m + s01 / n

    diferencia = auc_a - auc_b
    varianza = covarianza[0, 0] + covarianza[1, 1] - 2 * covarianza[0, 1]
    z = diferencia / np.sqrt(varianza) if varianza > 0 else 0.0
    return {"auc_a": auc_a, "auc_b": auc_b, "diferencia": diferencia,
            "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


# ==========================================================================
# Comparaciones múltiples
# ==========================================================================

def correccion_holm(valores_p):
    """Corrección de Holm-Bonferroni para comparaciones múltiples."""
    p = np.asarray(valores_p, dtype=float)
    m = len(p)
    orden = np.argsort(p)
    ajustados, maximo = np.empty(m), 0.0
    for i, indice in enumerate(orden):
        maximo = max(maximo, (m - i) * p[indice])
        ajustados[indice] = min(1.0, maximo)
    return ajustados


# ==========================================================================
# Intervalos de confianza
# ==========================================================================

def bootstrap_bca(y_real, probabilidades, grupos, funcion_metrica,
                  n_repeticiones=2_000, confianza=0.95, semilla=SEMILLA):
    """Intervalo de confianza BCa remuestreando grupos completos.

    Parameters
    ----------
    y_real, probabilidades : array-like
    grupos : array-like
        Identificador de paciente; el remuestreo toma pacientes completos.
    funcion_metrica : callable
        Recibe (y_real, probabilidades) y devuelve un escalar.
    n_repeticiones : int
    confianza : float
    semilla : int

    Returns
    -------
    dict
        Estimación puntual y límites del intervalo.
    """
    aleatorio = np.random.default_rng(semilla)
    y_real = np.asarray(y_real)
    probabilidades = np.asarray(probabilidades)
    unicos = np.unique(grupos)
    indices_por_grupo = {g: np.where(grupos == g)[0] for g in unicos}

    estimacion = funcion_metrica(y_real, probabilidades)

    replicas = []
    for _ in range(n_repeticiones):
        muestra = aleatorio.choice(unicos, size=len(unicos), replace=True)
        indices = np.concatenate([indices_por_grupo[g] for g in muestra])
        if len(np.unique(y_real[indices])) < 2:
            continue
        replicas.append(funcion_metrica(y_real[indices],
                                        probabilidades[indices]))
    replicas = np.asarray(replicas)

    # Corrección de sesgo: proporción de réplicas por debajo de la
    # estimación original, transformada a la escala normal.
    proporcion = np.mean(replicas < estimacion)
    proporcion = np.clip(proporcion, 1e-6, 1 - 1e-6)
    z0 = stats.norm.ppf(proporcion)

    # Aceleración por jackknife sobre los grupos.
    jackknife = []
    for g in unicos:
        indices = np.concatenate([idx for gg, idx in indices_por_grupo.items()
                                  if gg != g])
        if len(np.unique(y_real[indices])) < 2:
            continue
        jackknife.append(funcion_metrica(y_real[indices],
                                         probabilidades[indices]))
    jackknife = np.asarray(jackknife)
    desviaciones = jackknife.mean() - jackknife
    denominador = 6 * (np.sum(desviaciones ** 2) ** 1.5)
    aceleracion = (np.sum(desviaciones ** 3) / denominador
                   if denominador > 0 else 0.0)

    alfa = 1 - confianza
    inferior, superior = [], []
    for z_alfa in [stats.norm.ppf(alfa / 2), stats.norm.ppf(1 - alfa / 2)]:
        ajustado = z0 + (z0 + z_alfa) / (1 - aceleracion * (z0 + z_alfa))
        percentil = 100 * stats.norm.cdf(ajustado)
        (inferior if z_alfa < 0 else superior).append(
            np.percentile(replicas, np.clip(percentil, 0, 100)))

    return {"estimación": estimacion, "IC inferior": inferior[0],
            "IC superior": superior[0], "réplicas válidas": len(replicas)}


# ==========================================================================
# Tamaño del efecto
# ==========================================================================

def delta_cliff(a, b):
    """Delta de Cliff entre dos muestras a partir del estadístico U."""
    u = stats.mannwhitneyu(a, b).statistic
    return 2 * u / (len(a) * len(b)) - 1


def magnitud_cliff(delta):
    """Clasifica la delta de Cliff según los umbrales convencionales."""
    d = abs(delta)
    if d < 0.147:
        return "Insignificante"
    if d < 0.330:
        return "Pequeño"
    if d < 0.474:
        return "Mediano"
    return "Grande"


def d_cohen(a, b, corregir=True):
    """d de Cohen con varianza combinada y corrección de Hedges.

    Trata las dos muestras como independientes, igual que la delta de Cliff
    del capítulo 11, para que ambas medidas sean comparables. Con muestras
    pequeñas, d sobrestima el efecto; el factor
    ``1 - 3 / (4 (n_a + n_b) - 9)`` lo corrige (g de Hedges).

    Returns
    -------
    float
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    varianza = (((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1))
                / (n_a + n_b - 2))
    if varianza == 0:
        return float("nan")
    d = (a.mean() - b.mean()) / np.sqrt(varianza)
    if corregir:
        d *= 1 - 3 / (4 * (n_a + n_b) - 9)
    return float(d)


def d_pareada(a, b):
    """d_z: media de las diferencias pareadas sobre su desviación.

    Es la versión coherente con el diseño pareado: los modelos se evalúan
    sobre los mismos folds externos, así que la variabilidad común a un fold
    se cancela en la diferencia.
    """
    diferencias = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = diferencias.std(ddof=1)
    return float(diferencias.mean() / sd) if sd > 0 else float("nan")


def magnitud_cohen(d):
    """Umbrales convencionales de Cohen: 0.2, 0.5 y 0.8."""
    if np.isnan(d):
        return "Indefinida"
    d = abs(d)
    if d < 0.2:
        return "Insignificante"
    if d < 0.5:
        return "Pequeño"
    if d < 0.8:
        return "Mediano"
    return "Grande"


def tabla_cohen(por_fold):
    """d de Cohen y d_z para todos los pares de columnas.

    Parameters
    ----------
    por_fold : pandas.DataFrame
        Filas: folds externos. Columnas: modelos.

    Returns
    -------
    pandas.DataFrame
    """
    filas = []
    for a, b in combinations(por_fold.columns, 2):
        d = d_cohen(por_fold[a], por_fold[b])
        dz = d_pareada(por_fold[a], por_fold[b])
        filas.append({
            "modelo A": a, "modelo B": b,
            "AUC-PR medio A": float(por_fold[a].mean()),
            "AUC-PR medio B": float(por_fold[b].mean()),
            "diferencia": float(por_fold[a].mean() - por_fold[b].mean()),
            "d de Cohen (Hedges)": d, "magnitud d": magnitud_cohen(d),
            "d_z pareada": dz, "magnitud d_z": magnitud_cohen(dz),
        })
    return pd.DataFrame(filas)


def prueba_nadeau_bengio(diferencias, razon_prueba_entrenamiento=0.25,
                         alternativa="mayor"):
    """t corregida de Nadeau y Bengio (2003) para validación cruzada.

    Los pliegues comparten entrenamiento, así que las diferencias no son
    independientes y la t ordinaria subestima la varianza. La corrección
    multiplica la varianza por ``1/k + n_prueba/n_entrenamiento``; con 5
    pliegues, ``n_prueba/n_entrenamiento = 1/4``.

    Returns
    -------
    dict
        ``t``, grados de libertad y valor p (unilateral si
        ``alternativa="mayor"``: H1, la media de las diferencias es > 0).
    """
    d = np.asarray(diferencias, dtype=float)
    k = len(d)
    varianza = d.var(ddof=1)
    if varianza == 0:
        t = np.inf if d.mean() > 0 else (-np.inf if d.mean() < 0 else 0.0)
    else:
        t = d.mean() / np.sqrt((1 / k + razon_prueba_entrenamiento) * varianza)
    if alternativa == "mayor":
        p = float(stats.t.sf(t, k - 1))
    else:
        p = float(2 * stats.t.sf(abs(t), k - 1))
    return {"t": float(t), "gl": k - 1, "p": p}


# ==========================================================================
# Pruebas complementarias (añadidas después del protocolo pre-registrado)
# ==========================================================================

def prueba_mcnemar_exacta(y_real, decision_a, decision_b):
    """Prueba exacta de McNemar (1947) sobre las decisiones de dos clasificadores.

    Solo cuentan los encuentros *discordantes*: los que un modelo clasifica
    bien y el otro mal. Si los dos modelos fueran igual de buenos, cada
    discordante caería de un lado o del otro con probabilidad 1/2, así que
    el número ``n01`` sigue una binomial(n01 + n10, 1/2). La versión exacta
    usa esa binomial en lugar de la aproximación chi-cuadrado, que falla
    con pocos discordantes.

    Parameters
    ----------
    y_real : array (n,)
        Etiquetas 0/1.
    decision_a, decision_b : array (n,)
        Clases predichas (0/1 o booleanas) por cada modelo sobre los mismos
        encuentros.

    Returns
    -------
    dict
        ``n01`` (a acierta y b falla), ``n10`` (a falla y b acierta) y ``p``
        bilateral.
    """
    y = np.asarray(y_real).astype(bool)
    acierta_a = np.asarray(decision_a).astype(bool) == y
    acierta_b = np.asarray(decision_b).astype(bool) == y
    n01 = int((acierta_a & ~acierta_b).sum())
    n10 = int((~acierta_a & acierta_b).sum())
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    return {"n01": n01, "n10": n10, "p": float(p)}


def t_bayesiana_correlacionada(diferencias, rope=0.005,
                               razon_prueba_entrenamiento=0.25):
    """Prueba t bayesiana correlacionada (Corani y Benavoli, 2015).

    Es la versión bayesiana de la t corregida de Nadeau y Bengio: con una
    priori no informativa, la posterior de la diferencia media es una t de
    Student con ``k - 1`` grados de libertad, centrada en la media de las
    diferencias por fold y con escala ``sqrt((1/k + rho/(1-rho)) * s^2)``,
    donde ``rho/(1-rho)`` corrige la correlación entre folds que comparten
    entrenamiento (con 5 folds, ``rho = 1/5`` y ``rho/(1-rho) = 1/4``).

    En lugar de un valor p devuelve tres probabilidades posteriores, usando
    una región de equivalencia práctica (ROPE) de ``±rope``: que el primer
    modelo sea mejor (diferencia > rope), equivalente (|diferencia| <= rope)
    o peor (diferencia < -rope).

    Parameters
    ----------
    diferencias : array (k,)
        Métrica del modelo A menos la del modelo B en cada fold.
    rope : float
        Mitad del ancho de la región de equivalencia práctica.
    razon_prueba_entrenamiento : float
        ``rho / (1 - rho)``; 1/4 con cinco folds.

    Returns
    -------
    dict
        ``mejor``, ``equivalente``, ``peor``, y la media y escala de la
        posterior.
    """
    d = np.asarray(diferencias, dtype=float)
    k = len(d)
    escala = np.sqrt((1 / k + razon_prueba_entrenamiento) * d.var(ddof=1))
    if escala == 0:
        media = d.mean()
        return {"mejor": float(media > rope), "equivalente": float(abs(media) <= rope),
                "peor": float(media < -rope), "media": float(media), "escala": 0.0}
    posterior = stats.t(k - 1, loc=d.mean(), scale=escala)
    mejor = float(posterior.sf(rope))
    peor = float(posterior.cdf(-rope))
    return {"mejor": mejor, "equivalente": 1.0 - mejor - peor, "peor": peor,
            "media": float(d.mean()), "escala": float(escala)}
