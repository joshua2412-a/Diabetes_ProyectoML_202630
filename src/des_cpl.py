"""
DES-CPL: selección dinámica de ensambles por competencia probabilística local.

Modelo original del Entregable 3. Para cada paciente nuevo decide cuánto
confiar en cada clasificador base según la calidad de sus probabilidades en
pacientes parecidos. Tiene cuatro componentes (sección de metodología del
artículo, ``sn-article.tex``):

1. **Conjunto de selección fuera de pliegue.** Cada modelo base se reentrena
   en una validación cruzada agrupada por paciente y se guarda, para cada
   encuentro, la probabilidad del modelo que no lo vio. Sobre los encuentros
   de entrenamiento las probabilidades serían optimistas y todos los modelos
   parecerían competentes.
2. **Región de competencia estratificada y agrupada.** Los ``k`` vecinos
   positivos y los ``k`` negativos más cercanos en el espacio de perfiles de
   salida (logit de las probabilidades de los ``M`` modelos), excluyendo los
   encuentros del mismo paciente cuando la consulta pertenece al conjunto de
   selección. Con un 11 % de positivos, una región sin estratificar suele no
   tener ningún positivo.
3. **Competencia por Brier local.** La puntuación de Brier de cada modelo
   en la región, promediada por clase y reponderada con la prevalencia local
   estimada (estimador estratificado del riesgo de Brier en esa zona). Mide
   la calidad de la probabilidad y no solo el acierto de la etiqueta. La
   versión con peso 1/2 por clase (Brier balanceado) queda disponible con
   ``ponderacion="balanceada"``, pero no es una regla propia para el riesgo
   local: premia al modelo cuya probabilidad está más cerca de 0.5, que en
   este problema es casi siempre el de probabilidad más alta.
4. **Ponderación softmax con temperatura.** ``w_m = softmax(-C_m / tau)``.
   Con ``tau`` muy grande es el promedio simple; con ``tau`` cero, la
   selección del modelo localmente mejor.

Dos componentes se añadieron después del piloto (capítulo 13), porque el
diagnóstico mostró que la competencia local es una estimación ruidosa:

5. **Encogimiento hacia la competencia global.** ``(1 - λ) C_local + λ
   C_global``, con λ elegido en el conjunto de selección. λ = 0 es la
   selección dinámica pura; λ = 1, pesos globales. Que los datos elijan λ
   es también una prueba: si eligen λ ≈ 1, no hay heterogeneidad local
   aprovechable.
6. **Poda del conjunto.** Solo se combinan los modelos con mayor AUC-PR
   fuera de pliegue; los débiles añaden ruido a la región y a la mezcla.

El módulo separa el mecanismo (funciones que operan sobre probabilidades ya
calculadas) del estimador compatible con scikit-learn (``DESCPL``). Así la
búsqueda de ``k`` y ``tau`` evalúa cada configuración sobre las
predicciones fuera de pliegue sin reentrenar los modelos base, y el estudio
de ablación cambia un componente con un argumento.

Implementación propia: solo se usan de scikit-learn la búsqueda de vecinos
(``StratifiedGroupKFold`` y ``clone``) y scipy para el índice de vecinos
(``cKDTree``). El mecanismo (competencia, encogimiento, poda y
combinación) está programado con numpy.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from scipy.spatial import cKDTree
from sklearn.utils.validation import check_is_fitted

from src import configuracion as cfg

#: Recorte de las probabilidades antes del logit: el árbol y Naive Bayes
#: producen ceros y unos exactos.
EPSILON = 1e-6

COMPETENCIAS = ("brier", "exactitud")
PONDERACIONES = ("prevalencia", "balanceada")

#: Pseudo-observaciones con que la prevalencia local se suaviza hacia la
#: global.
SUAVIZADO_PREVALENCIA = 10


# ---------------------------------------------------------------------------
# Componentes del mecanismo
# ---------------------------------------------------------------------------

def perfil(probabilidades):
    """Perfil de salida: logit de las probabilidades recortadas, (n, M)."""
    p = np.clip(np.asarray(probabilidades, dtype=float), EPSILON, 1 - EPSILON)
    return np.log(p) - np.log1p(-p)


def pesos_softmax(competencias, tau):
    """Pesos por consulta a partir de competencias (menor es mejor).

    ``tau = inf`` da pesos iguales y ``tau = 0`` reparte todo el peso entre
    los modelos de menor competencia (empates a partes iguales). Se resta el
    mínimo por fila antes de exponenciar para evitar desbordamientos.
    """
    c = np.asarray(competencias, dtype=float)
    if np.isinf(tau):
        return np.full_like(c, 1.0 / c.shape[1])
    if tau <= 0:
        mejores = np.isclose(c, c.min(axis=1, keepdims=True), rtol=0, atol=1e-12)
        return mejores / mejores.sum(axis=1, keepdims=True)
    z = np.exp(-(c - c.min(axis=1, keepdims=True)) / tau)
    return z / z.sum(axis=1, keepdims=True)


def combinar(pesos, probabilidades):
    """Probabilidad del ensamble: sum_m w_m(x) f_m(x)."""
    return np.sum(np.asarray(pesos) * np.asarray(probabilidades), axis=1)


def competencia_local(prob_vecinos, y_vecinos, tipo="brier", peso_positivos=0.5):
    """Competencia de cada modelo en la región de cada consulta.

    ``prob_vecinos`` tiene forma (n_consultas, n_vecinos, M) y ``y_vecinos``
    (n_consultas, n_vecinos). La pérdida de cada clase se promedia por
    separado y se combinan como ``q * pérdida_positivos + (1 - q) *
    pérdida_negativos``, con ``q = peso_positivos`` (escalar o un valor por
    consulta). Si una región no tiene vecinos de una clase, se usa solo la
    otra.

    * ``q = 0.5`` es el Brier balanceado.
    * ``q`` igual a la prevalencia local estimada es el estimador
      estratificado del riesgo de Brier local: la región garantiza vecinos
      de las dos clases, y la reponderación devuelve a cada clase su peso
      real en esa zona.

    ``tipo="brier"`` es la medida propuesta; ``tipo="exactitud"`` es la
    tasa de error con umbral 0.5 (ablación 1).
    """
    if tipo not in COMPETENCIAS:
        raise ValueError(f"Competencia desconocida: {tipo!r}; usa {COMPETENCIAS}")
    y = np.asarray(y_vecinos, dtype=float)[:, :, None]
    p = np.asarray(prob_vecinos, dtype=float)
    perdida = (p - y) ** 2 if tipo == "brier" else ((p >= 0.5) != (y == 1))
    positivos = y == 1
    n_pos = positivos.sum(axis=1)
    n_neg = (~positivos).sum(axis=1)
    media_pos = np.where(positivos, perdida, 0).sum(axis=1) / np.maximum(n_pos, 1)
    media_neg = np.where(~positivos, perdida, 0).sum(axis=1) / np.maximum(n_neg, 1)
    q = np.broadcast_to(np.asarray(peso_positivos, dtype=float).reshape(-1, 1)
                        if np.ndim(peso_positivos) else peso_positivos,
                        media_pos.shape)
    q = np.where(n_pos == 0, 0.0, np.where(n_neg == 0, 1.0, q))
    return q * media_pos + (1 - q) * media_neg


class RegionCompetencia:
    """Índices de vecinos sobre el conjunto de selección.

    Mantiene un índice por clase (para la región estratificada) y uno con
    todos los encuentros (para la prevalencia local y la región sin
    estratificar). La exclusión del paciente se hace pidiendo al índice
    ``k`` más tantos vecinos extra como encuentros tiene el paciente más
    frecuente, y descartando después los del mismo paciente: así siempre
    quedan al menos ``k`` válidos.
    """

    def __init__(self, perfiles, y, grupos=None):
        self.perfiles = np.asarray(perfiles, dtype=float)
        self.y = np.asarray(y).astype(int)
        self.grupos = None if grupos is None else np.asarray(grupos)
        self.prevalencia = float(self.y.mean())
        self._indices = {}
        for clase in (0, 1, None):
            filas = (np.flatnonzero(self.y == clase) if clase is not None
                     else np.arange(len(self.y)))
            if len(filas) == 0:
                raise ValueError(f"El conjunto de selección no tiene la clase {clase}")
            extra = 0
            if self.grupos is not None:
                _, cuentas = np.unique(self.grupos[filas], return_counts=True)
                extra = int(cuentas.max())
            arbol = cKDTree(self.perfiles[filas])
            self._indices[clase] = (filas, arbol, extra)

    def vecinos(self, perfiles_consulta, k, grupos_consulta=None, clase=None):
        """Los ``k`` vecinos más cercanos de cada consulta, de una clase o de todas.

        Devuelve índices del conjunto de selección, (n_consultas, k),
        ordenados por cercanía. Si ``grupos_consulta`` se da, ningún vecino
        comparte paciente con su consulta.
        """
        consultas = np.asarray(perfiles_consulta, dtype=float)
        excluir = grupos_consulta is not None and self.grupos is not None
        filas, arbol, extra = self._indices[clase]
        pedir = min(len(filas), k + (extra if excluir else 0))
        if pedir < k:
            raise ValueError(f"k={k} supera los encuentros disponibles")
        _, posiciones = arbol.query(consultas, k=pedir)
        posiciones = np.asarray(posiciones).reshape(len(consultas), pedir)
        candidatos = filas[posiciones]
        if not excluir:
            return candidatos[:, :k]
        validos = self.grupos[candidatos] != np.asarray(grupos_consulta)[:, None]
        # El orden estable conserva la cercanía entre los válidos.
        orden = np.argsort(~validos, axis=1, kind="stable")[:, :k]
        if not np.take_along_axis(validos, orden, axis=1).all():
            raise ValueError("No quedan vecinos suficientes tras excluir al paciente")
        return np.take_along_axis(candidatos, orden, axis=1)

    def region(self, perfiles_consulta, k, grupos_consulta=None,
               estratificado=True):
        """Región de competencia: ``k`` negativos y ``k`` positivos, o ``2k`` sin distinguir."""
        if not estratificado:
            return self.vecinos(perfiles_consulta, 2 * k, grupos_consulta)
        return np.hstack([self.vecinos(perfiles_consulta, k, grupos_consulta, 0),
                          self.vecinos(perfiles_consulta, k, grupos_consulta, 1)])

    def prevalencia_local(self, perfiles_consulta, k, grupos_consulta=None,
                          suavizado=SUAVIZADO_PREVALENCIA):
        """Proporción de positivos entre los ``2k`` vecinos sin estratificar.

        Se suaviza hacia la prevalencia global con ``suavizado``
        pseudo-observaciones, para que una zona con pocos vecinos no reciba
        un peso de 0 o 1.
        """
        idx = self.vecinos(perfiles_consulta, 2 * k, grupos_consulta)
        return _prevalencia_suavizada(self.y[idx], self.prevalencia, suavizado)

    def vecindario(self, perfiles_consulta, k, grupos_consulta=None):
        """Todos los vecinos que DES-CPL puede necesitar con hasta ``k`` por clase.

        Devuelve ``{0: (n, k), 1: (n, k), None: (n, 2k)}``. Como cada matriz
        está ordenada por cercanía, sus primeras columnas son el vecindario
        de cualquier ``k`` menor: la búsqueda de hiperparámetros consulta el
        índice una sola vez y recorta.
        """
        return {0: self.vecinos(perfiles_consulta, k, grupos_consulta, 0),
                1: self.vecinos(perfiles_consulta, k, grupos_consulta, 1),
                None: self.vecinos(perfiles_consulta, 2 * k, grupos_consulta)}


def _prevalencia_suavizada(y_vecinos, prevalencia, suavizado):
    return ((y_vecinos.sum(axis=1) + suavizado * prevalencia)
            / (y_vecinos.shape[1] + suavizado))


def competencia_global(prob_seleccion, y_seleccion, competencia="brier",
                       ponderacion="prevalencia"):
    """Competencia de cada modelo en todo el conjunto de selección, (1, M).

    Es la misma medida que la local, calculada sin vecindario: el
    estimador global hacia el que se encoge la competencia local.
    """
    y = np.asarray(y_seleccion)
    p = np.asarray(prob_seleccion, dtype=float)[None]
    q = 0.5 if ponderacion == "balanceada" else float(y.mean())
    return competencia_local(p, y[None], competencia, q)


def podar(prob_seleccion, y_seleccion, n_modelos):
    """Índices de los ``n_modelos`` con mayor AUC-PR fuera de pliegue.

    La poda se decide con el conjunto de selección, nunca con la evaluación.
    Con ``n_modelos=None`` se conservan todos.
    """
    from sklearn.metrics import average_precision_score

    M = np.asarray(prob_seleccion).shape[1]
    if n_modelos is None or n_modelos >= M:
        return np.arange(M)
    puntajes = [average_precision_score(y_seleccion, np.asarray(prob_seleccion)[:, m])
                for m in range(M)]
    # Orden estable: en empate se queda el primero, para que sea determinista.
    return np.sort(np.argsort(-np.asarray(puntajes), kind="stable")[:n_modelos])


def competencias_locales(region, vecindario, prob_seleccion, k,
                         competencia="brier", estratificado=True,
                         ponderacion="prevalencia",
                         suavizado=SUAVIZADO_PREVALENCIA,
                         encogimiento=0.0):
    """Competencias de cada modelo a partir de un vecindario ya calculado.

    Devuelve ``(competencias, indices_region, peso_positivos)``.
    ``ponderacion`` decide el peso de cada clase en la región
    estratificada: ``"prevalencia"`` (prevalencia local estimada con los
    ``2k`` vecinos sin estratificar) o ``"balanceada"`` (1/2 cada una).

    ``encogimiento`` (λ en [0, 1]) mezcla la competencia local con la
    global: ``(1 - λ) C_local + λ C_global``. Con pocos vecinos la
    estimación local es ruidosa; el encogimiento es el estimador de tipo
    James-Stein que la acerca a la global. λ = 0 es la competencia local
    pura y λ = 1 equivale a pesos globales.
    """
    if not 0.0 <= encogimiento <= 1.0:
        raise ValueError("encogimiento debe estar entre 0 y 1")
    if ponderacion not in PONDERACIONES:
        raise ValueError(f"Ponderación desconocida: {ponderacion!r}; usa {PONDERACIONES}")
    todos = vecindario[None][:, :2 * k]
    if not estratificado:
        # Muestra sin estratificar: cada clase pesa lo que pesa en la región.
        idx, q = todos, region.y[todos].mean(axis=1)
    else:
        idx = np.hstack([vecindario[0][:, :k], vecindario[1][:, :k]])
        q = (0.5 if ponderacion == "balanceada" else
             _prevalencia_suavizada(region.y[todos], region.prevalencia, suavizado))
    c = competencia_local(np.asarray(prob_seleccion)[idx], region.y[idx],
                          competencia, q)
    if encogimiento > 0:
        g = competencia_global(prob_seleccion, region.y, competencia, ponderacion)
        c = (1 - encogimiento) * c + encogimiento * g
    return c, idx, q


def predecir_con_seleccion(region, prob_seleccion, prob_consulta,
                           perfiles_consulta, k, tau, grupos_consulta=None,
                           competencia="brier", estratificado=True,
                           ponderacion="prevalencia", encogimiento=0.0,
                           detalle=False):
    """Aplica DES-CPL a consultas cuyas probabilidades ya se conocen.

    ``prob_seleccion`` son las probabilidades fuera de pliegue del conjunto
    de selección (n, M); ``prob_consulta`` las de los modelos que se
    combinan (n_consultas, M); ``perfiles_consulta`` la posición de cada
    consulta en el espacio de perfiles. ``ponderacion`` decide el peso de
    cada clase en la región estratificada: ``"prevalencia"`` (prevalencia
    local estimada) o ``"balanceada"`` (1/2 cada una). Con ``detalle=True``
    devuelve también competencias, pesos y vecinos.
    """
    vecindario = region.vecindario(perfiles_consulta, k, grupos_consulta)
    c, idx, q = competencias_locales(region, vecindario, prob_seleccion, k,
                                     competencia, estratificado, ponderacion,
                                     encogimiento=encogimiento)
    w = pesos_softmax(c, tau)
    p = combinar(w, prob_consulta)
    if detalle:
        return p, {"competencias": c, "pesos": w, "vecinos": idx,
                   "peso_positivos": np.broadcast_to(q, len(p))}
    return p


# ---------------------------------------------------------------------------
# Estimador
# ---------------------------------------------------------------------------

class DESCPL(ClassifierMixin, BaseEstimator):
    """Ensamble dinámico por competencia probabilística local.

    Parámetros
    ----------
    estimadores : list of (str, estimator)
        Modelos base con sus hiperparámetros ya fijados (por ejemplo, los
        pipelines del benchmark con su configuración modal).
    k : int
        Vecinos por clase en la región de competencia.
    tau : float
        Temperatura de la softmax, en unidades de la puntuación de Brier.
    n_pliegues : int
        Pliegues de la validación cruzada que produce las predicciones fuera
        de pliegue.
    competencia : {"brier", "exactitud"}
    estratificado : bool
        Región con ``k`` vecinos por clase (True) o ``2k`` sin distinguir.
    ponderacion : {"prevalencia", "balanceada"}
        Peso de cada clase en la región estratificada (ver
        ``competencia_local``).
    excluir_paciente : bool
        Excluir de la región los encuentros del mismo paciente al puntuar el
        conjunto de selección (``puntuar_seleccion``).
    encogimiento : float
        λ en [0, 1]: peso de la competencia global en la mezcla con la local.
    n_modelos : int or None
        Poda: se combinan solo los ``n_modelos`` con mayor AUC-PR fuera de
        pliegue (None: todos). El perfil de salida usa solo esos modelos.
    semilla : int
    """

    def __init__(self, estimadores, k=20, tau=0.01, n_pliegues=3,
                 competencia="brier", estratificado=True,
                 ponderacion="prevalencia", excluir_paciente=True,
                 encogimiento=0.0, n_modelos=None, semilla=cfg.SEMILLA):
        self.estimadores = estimadores
        self.k = k
        self.tau = tau
        self.n_pliegues = n_pliegues
        self.competencia = competencia
        self.estratificado = estratificado
        self.ponderacion = ponderacion
        self.excluir_paciente = excluir_paciente
        self.encogimiento = encogimiento
        self.n_modelos = n_modelos
        self.semilla = semilla

    def _lista_estimadores(self):
        """Modelos base que se combinan.

        Es un método, y no un acceso directo al atributo, para que una
        subclase pueda dar modelos por defecto sin modificar el parámetro
        ``estimadores`` (scikit-learn exige que ``fit`` no altere los
        hiperparámetros); lo usa ``src.original_model.ModeloOriginal``.
        """
        return self.estimadores

    def _particiones(self, X, y, grupos):
        if grupos is None:
            cv = StratifiedKFold(self.n_pliegues, shuffle=True,
                                 random_state=self.semilla)
            return cv.split(X, y)
        cv = StratifiedGroupKFold(self.n_pliegues, shuffle=True,
                                  random_state=self.semilla)
        return cv.split(X, y, grupos)

    def _validar(self, X, y, grupos):
        """Rechaza hiperparámetros y entradas inválidos antes de entrenar.

        Los valores faltantes en ``X`` no se rechazan aquí: los modelos base
        son pipelines que los imputan (capítulo 6). Lo que DES-CPL necesita
        de los datos es una etiqueta binaria, tamaños coherentes y
        probabilidades finitas, y lo verifica.
        """
        if not self._lista_estimadores():
            raise ValueError("estimadores no puede estar vacío")
        if not isinstance(self.k, (int, np.integer)) or self.k < 1:
            raise ValueError(f"k debe ser un entero positivo (recibido {self.k!r})")
        if not (self.tau >= 0):
            raise ValueError(f"tau debe ser >= 0 (recibido {self.tau!r})")
        if not 0.0 <= self.encogimiento <= 1.0:
            raise ValueError("encogimiento debe estar entre 0 y 1")
        if self.n_modelos is not None and not 1 <= self.n_modelos <= len(self._lista_estimadores()):
            raise ValueError("n_modelos debe estar entre 1 y el número de estimadores")
        if self.competencia not in COMPETENCIAS:
            raise ValueError(f"competencia debe ser una de {COMPETENCIAS}")
        if self.ponderacion not in PONDERACIONES:
            raise ValueError(f"ponderacion debe ser una de {PONDERACIONES}")
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("X e y deben tener el mismo número de filas")
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("DES-CPL es binario: y debe contener solo 0 y 1")
        if len(np.unique(y)) < 2:
            raise ValueError("y debe contener las dos clases")
        if grupos is not None and len(grupos) != len(y):
            raise ValueError("grupos debe tener una entrada por fila")
        return y.astype(int)

    def fit(self, X, y, grupos=None):
        """Ajusta modelos de pliegue, conjunto de selección y modelos finales."""
        y = self._validar(X, y, grupos)
        self.classes_ = np.array([0, 1])
        self.nombres_ = [nombre for nombre, _ in self._lista_estimadores()]
        n, M = len(y), len(self._lista_estimadores())

        self.prob_seleccion_ = np.full((n, M), np.nan)
        self.modelos_pliegue_ = []
        for entrenamiento, validacion in self._particiones(X, y, grupos):
            modelos = []
            for m, (_, estimador) in enumerate(self._lista_estimadores()):
                modelo = clone(estimador).fit(_filas(X, entrenamiento),
                                              y[entrenamiento])
                self.prob_seleccion_[validacion, m] = (
                    modelo.predict_proba(_filas(X, validacion))[:, 1])
                modelos.append(modelo)
            self.modelos_pliegue_.append(modelos)

        if not np.isfinite(self.prob_seleccion_).all():
            raise ValueError("Algún modelo base produjo probabilidades no finitas")
        self.y_seleccion_ = y
        self.grupos_seleccion_ = None if grupos is None else np.asarray(grupos)
        self.activos_ = podar(self.prob_seleccion_, y, self.n_modelos)
        self.region_ = RegionCompetencia(perfil(self._seleccion()), y,
                                         self.grupos_seleccion_)
        self.modelos_ = [clone(estimador).fit(X, y)
                         for _, estimador in self._lista_estimadores()]
        return self

    def _seleccion(self):
        return self.prob_seleccion_[:, self.activos_]

    def probabilidades_base(self, X):
        """Probabilidades de los modelos finales y perfil de las consultas.

        El perfil usa el promedio de los modelos de pliegue, entrenados con
        el mismo tamaño de muestra que produjo el conjunto de selección, para
        que consultas y vecinos sean comparables.
        """
        finales = np.column_stack([self.modelos_[m].predict_proba(X)[:, 1]
                                   for m in self.activos_])
        de_pliegue = np.mean([
            np.column_stack([modelos[m].predict_proba(X)[:, 1] for m in self.activos_])
            for modelos in self.modelos_pliegue_], axis=0)
        return finales, perfil(de_pliegue)

    def predict_proba(self, X, detalle=False):
        check_is_fitted(self, "region_")
        finales, perfiles = self.probabilidades_base(X)
        resultado = predecir_con_seleccion(
            self.region_, self._seleccion(), finales, perfiles,
            self.k, self.tau, competencia=self.competencia,
            estratificado=self.estratificado, ponderacion=self.ponderacion,
            encogimiento=self.encogimiento, detalle=detalle)
        p = resultado[0] if detalle else resultado
        proba = np.column_stack([1 - p, p])
        return (proba, resultado[1]) if detalle else proba

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def puntuar_seleccion(self, k=None, tau=None, encogimiento=None, metrica=None):
        """Desempeño de una configuración sobre el conjunto de selección.

        Cada encuentro se predice con su probabilidad fuera de pliegue y una
        región que excluye a su propio paciente (si ``excluir_paciente``).
        No reentrena nada, así que la búsqueda de ``k`` y ``tau`` cuesta solo
        consultas de vecinos. Devuelve la métrica (por defecto, AUC-PR).
        """
        from sklearn.metrics import average_precision_score

        metrica = metrica or average_precision_score
        k = self.k if k is None else k
        tau = self.tau if tau is None else tau
        encogimiento = self.encogimiento if encogimiento is None else encogimiento
        grupos = self.grupos_seleccion_ if self.excluir_paciente else None
        if grupos is None:
            # Sin grupos, cada encuentro es su propio paciente: se excluye a
            # sí mismo para no evaluarse con su propia etiqueta.
            grupos = np.arange(len(self.y_seleccion_))
            region = RegionCompetencia(self.region_.perfiles, self.y_seleccion_,
                                       grupos)
        else:
            region = self.region_
        seleccion = self._seleccion()
        p = predecir_con_seleccion(region, seleccion, seleccion, region.perfiles,
                                   k, tau, grupos_consulta=grupos,
                                   competencia=self.competencia,
                                   estratificado=self.estratificado,
                                   ponderacion=self.ponderacion,
                                   encogimiento=encogimiento)
        return float(metrica(self.y_seleccion_, p))


def _filas(X, indices):
    return X.iloc[indices] if hasattr(X, "iloc") else X[indices]
