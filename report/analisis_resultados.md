# Análisis de resultados

Este documento resume la evidencia de `results/runs.csv`, `results/summary.csv`, `results/final_metrics.csv` y las curvas generadas el 2026-06-10.

## Metodología

- Dataset binario `cornell-movie-review-data/rotten_tomatoes`.
- Splits oficiales: 8530 train, 1066 validación y 1066 test.
- TF-IDF adaptado exclusivamente con train.
- Diez semillas por configuración.
- Early stopping sobre F1 de validación.
- Umbral sigmoide ajustado exclusivamente con validación.
- Test consultado una sola vez para el ganador global.

Las diez configuraciones iniciales y las seis variantes SGD produjeron 160 corridas únicas.

## Hallazgos principales

### La representación fue más importante que aumentar neuronas

El baseline Adam de 32 neuronas obtuvo F1 medio 0.7651. Aumentar a 128 neuronas produjo 0.7682, con intervalos solapados. La MLP de 32 neuronas con unigramas, bigramas y TF-IDF normalizado alcanzó 0.7722 y fue el ganador.

El baseline lineal sin capa oculta llegó a 0.7697. Esto muestra que el espacio TF-IDF ya es fuertemente separable de forma lineal y explica por qué agregar capacidad aporta poco.

### La regularización no mejoró de manera universal

Dropout 0.2 fue competitivo con Adam y 128 neuronas, pero no estableció una ventaja concluyente. L2 quedó cerca o por debajo de las variantes sin regularización. Early stopping ya controla parte del sobreajuste y dropout puede eliminar demasiada señal en una red chica.

Por lo tanto, que la mejor red no use dropout ni L2 no es por sí solo evidencia de fuga de datos.

### SGD no superó a Adam

| Configuración SGD | F1 medio | Desvío |
| --- | ---: | ---: |
| LR 0.001, momentum 0.9 | **0.7576** | 0.0065 |
| LR 0.01, momentum 0.9 | 0.7554 | 0.0106 |
| LR 0.01, momentum 0.9, L2 | 0.7539 | 0.0130 |
| LR 0.01, sin momentum | 0.7476 | 0.0285 |
| LR 0.01, momentum 0.9, dropout 0.2 | 0.7370 | 0.0084 |
| LR 0.1, momentum 0.9 | 0.6889 | 0.0094 |

Momentum redujo la variabilidad respecto de SGD puro. El LR bajo fue la mejor variante, aunque necesitó 18.3 épocas en promedio. LR 0.1 fue demasiado agresivo y dropout perjudicó el aprendizaje.

### El umbral tiene un efecto medible

El ganador obtuvo F1 medio 0.7592 con umbral fijo 0.5 y 0.7722 al ajustar el corte en validación. Esa diferencia supera la observada entre 32 y 128 neuronas.

La capa final continúa siendo `Dense(1, activation="sigmoid")`; el ajuste solo decide en qué probabilidad se transforma la salida en clase.

## Evaluación final

El ganador fue `e9_mlp32_bigrams`. Reentrenado con semilla 42, su umbral de validación fue 0.4851.

| Métrica test | Valor |
| --- | ---: |
| Accuracy | 0.7824 |
| Precision | 0.7573 |
| Recall | 0.8311 |
| F1 | 0.7925 |

La matriz final contiene 391 verdaderos negativos, 142 falsos positivos, 90 falsos negativos y 443 verdaderos positivos. El modelo favorece recall positivo: recupera más críticas positivas a costa de aceptar falsos positivos.

## Overfitting

La simulación intencional usa 512 neuronas, 500 ejemplos, 35 épocas y no aplica early stopping. Train llega a accuracy 1.0 y loss casi cero, mientras validación queda cerca de 0.64 y su loss aumenta. Esta corrida demuestra memorización y se mantiene separada de la selección de hiperparámetros.

## Limitaciones

- Las semillas miden variación de inicialización y entrenamiento, pero usan el mismo split oficial.
- TF-IDF no conserva orden ni contexto completo.
- El umbral se optimiza para F1; otra aplicación podría priorizar precision o recall.
- No se realizó análisis cualitativo de ejemplos mal clasificados.

## Conclusión general

La evidencia no permite afirmar que 128 neuronas sean necesarias. La diferencia frente a 32 neuronas es pequeña y sus intervalos se solapan. Además, una red de 32 neuronas sobre 5000 entradas ya contiene 160 065 parámetros.

El cambio más útil ocurrió en la representación: la MLP de 32 neuronas con bigramas y normalización fue superior al baseline de unigramas. El modelo lineal, con solo 5001 parámetros, quedó muy cerca del ganador, señal de que TF-IDF vuelve al problema mayormente separable de forma lineal.

Que el ganador no use dropout ni L2 no constituye evidencia de fuga. El vectorizador se adapta únicamente con train, la selección usa validación y test permanece reservado. Early stopping ya limita el sobreajuste; en una capa pequeña, dropout puede eliminar señal útil.

Adam fue más eficaz y estable que todas las variantes SGD probadas. Momentum redujo variabilidad, pero no cerró la diferencia. El siguiente paso razonable no es aumentar neuronas, sino separar el efecto de bigramas y normalización, modelar mejor negaciones y analizar errores individuales.

En términos académicos, la red es un baseline sólido y reproducible, con F1 test 0.7925. Sirve para demostrar correctamente el ciclo completo de preparación, entrenamiento, sobreajuste y evaluación, aunque no representa el techo del análisis de sentimiento.
