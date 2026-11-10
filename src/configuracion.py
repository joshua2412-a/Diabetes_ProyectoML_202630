"""
Configuración global compartida por todos los módulos.

Reúne la semilla, el número de procesos y las piezas comunes (la excepción
de combinaciones no aplicables y la importación de dependencias opcionales).

Forma parte de ``src/``, la versión modular del motor del experimento. Las
funciones se trasladaron sin cambios de lógica desde ``docs/experimento.py``,
que se conserva congelado para reproducir el Entregable 2; la equivalencia
entre ambos se verifica en ``tests/test_equivalencia_e2.py``.

``N_JOBS`` se lee siempre en el momento de usarlo, nunca al importar: los
demás módulos acceden a él como ``configuracion.N_JOBS``, de modo que basta
con cambiarlo una vez (``cfg.N_JOBS = 4``) antes de construir los pipelines.
"""

from __future__ import annotations

import importlib
import warnings

#: Semilla global única, propagada a numpy, scikit-learn, XGBoost, Optuna y
#: al algoritmo genético.
SEMILLA = 42

#: Procesos para la paralelización. -1 usa todos los núcleos lógicos. Puede
#: cambiarse desde un notebook (``ex.N_JOBS = 4``) antes de construir los
#: pipelines: todas las fábricas y el ejecutor leen este valor en el momento
#: de usarlo. En el equipo del proyecto (4 núcleos físicos, 8 lógicos), 8
#: procesos solo acortan la búsqueda un 9 % respecto a 4 y duplican la
#: memoria, así que el experimento del capítulo 7 se ejecuta con 4.
N_JOBS = -1

# scikit-learn 1.8 marca ``penalty`` como obsoleto en favor de ``l1_ratio`` y
# avisa en cada ajuste con penalty="l1". El comportamiento no cambia (el
# ajuste sigue siendo L1), así que el aviso solo añade ruido a la salida.
warnings.filterwarnings(
    "ignore", message="Inconsistent values: penalty=.* with l1_ratio",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message="'penalty' was deprecated", category=FutureWarning,
)


class CombinacionNoAplicable(ValueError):
    """Combinación metodológicamente imposible, no un error de ejecución.

    Se distingue de ``ValueError`` para que el ejecutor registre como
    "no aplicable" solo las combinaciones que lo son por diseño (k-NN con
    class_weight), y deje propagarse cualquier error real de programación
    o de datos en lugar de ocultarlo en la tabla maestra.
    """


def requiere(modulo, paquete=None):
    """Importa un módulo opcional con un mensaje de error accionable."""
    try:
        return importlib.import_module(modulo)
    except ImportError as exc:
        paquete = paquete or modulo
        raise ImportError(
            f"Este componente requiere '{paquete}'. Instálalo con "
            f"`pip install {paquete}` y vuelve a ejecutar la celda."
        ) from exc
