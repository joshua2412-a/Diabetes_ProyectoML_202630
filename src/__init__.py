"""
Motor del experimento de readmisión temprana en pacientes diabéticos.

Versión modular de ``docs/experimento.py``. Cada módulo tiene una sola
responsabilidad:

========================  ==================================================
``configuracion``         Semilla, número de procesos y piezas comunes
``preprocesamiento``      Preprocesador por bloques de variables
``modelos``               Los siete modelos base y sus catálogos
``espacios``              Rejillas, espacios continuos y presupuesto
``balanceo``              Estrategias de balanceo y pipeline completo
``metricas``              Métricas para clases desbalanceadas
``optimizacion``          Grid, Random, Optuna y algoritmo genético
``perfilamiento``         Tiempo y memoria de un ajuste
``validacion``            Validación anidada agrupada y ejecutor
``estadistica``           Pruebas de hipótesis y tamaños del efecto
``explicacion``           Valores de Shapley sobre las variables originales
``des_cpl``               Modelo original del Entregable 3 (DES-CPL)
========================  ==================================================

``docs/experimento.py`` se conserva congelado para reproducir el Entregable
2; ``tests/test_equivalencia_e2.py`` verifica que ambos dan los mismos
resultados. El código nuevo del Entregable 3 usa este paquete.
"""
