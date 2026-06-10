# Analisis de resultados (insumo tecnico para el informe)

Este documento interpreta la evidencia generada por `python src/train.py` el 2026-06-10: `results/metrics.csv`, `results/confusion_matrix.png`, `results/best_learning_curves.png` y `results/overfitting_learning_curves.png`. Todos los numeros citados salen de esos archivos y son reproducibles (semilla fija).

Es un insumo de trabajo: el informe final debe redactarse con palabras propias del grupo, siguiendo la estructura de `guia_informe.md`.

## 1. Dataset y configuracion general

- Dataset: `cornell-movie-review-data/rotten_tomatoes` (criticas de cine, sentimiento binario).
- Splits oficiales: 8530 ejemplos de train, 1066 de validacion y 1066 de test, balanceados (533 positivos y 533 negativos en test).
- Entrada: TF-IDF de 5000 tokens, adaptado solo con train.
- Todos los experimentos comparten batch size 32 y optimizador Adam; lo que varia cada uno esta en `src/config.py`.

## 2. Tabla comparativa (de `metrics.csv`, ordenada por val_f1)

| Experimento | Config distintiva | Epocas corridas | val_f1 | test_f1 | test_acc |
| --- | --- | --- | --- | --- | --- |
| e3_regularized | 128 neuronas + dropout 0.5 + L2 1e-4 | 6 de 12 | **0.7647** | 0.7775 | 0.7702 |
| e0_baseline | 32 neuronas | 6 de 12 | 0.7525 | 0.7662 | 0.7589 |
| e1_more_capacity | 128 neuronas | 6 de 12 | 0.7447 | 0.7656 | 0.7570 |
| e2_lower_learning_rate | lr 1e-4 (resto = baseline) | 12 de 12 | 0.7405 | 0.7584 | 0.7645 |
| e4_overfitting_demo | 512 neuronas, 500 ejemplos, 35 epocas, sin early stopping | 35 de 35 | 0.6127 | 0.6465 | 0.6604 |

El mejor por F1 de validacion es `e3_regularized`; el test se uso una sola vez, sobre ese modelo.

## 3. Lectura experimento por experimento

**e0_baseline.** Una red chica ya resuelve el problema de forma aceptable (val_f1 0.7525). Es el punto de referencia contra el que se mide todo lo demas.

**e1_more_capacity (vs e0).** Cuadruplicar las neuronas NO mejoro validacion (0.7447 < 0.7525). Mas parametros ajustan mas rapido el train, pero sobre una entrada TF-IDF de 5000 dimensiones tambien memorizan antes patrones poco generalizables. Es el resultado que motiva el experimento de regularizacion.

**e2_lower_learning_rate (vs e0).** Bajar el learning rate de 1e-3 a 1e-4 hizo el entrenamiento mas lento y estable: fue el unico experimento que agoto las 12 epocas sin que el early stopping cortara antes. Pero no gano en validacion (0.7405). Conclusion util para el informe: un cambio "mas prudente" no es automaticamente mejor; se compara empiricamente.

**e3_regularized (vs e1).** Misma capacidad que e1 (128 neuronas) pero con dropout 0.5 y L2 1e-4: pasa de 0.7447 a 0.7647 de val_f1, el mejor del lote. La regularizacion recupero la capacidad extra que e1 desperdiciaba en memorizar. Esta es la comparacion mas valiosa del TP: capacidad sin control empeora, capacidad regularizada gana.

**e4_overfitting_demo.** Ver seccion 5.

## 4. Un patron transversal: el sobreajuste aparece desde la primera epoca

Dato no obvio que conviene contar: e0, e1 y e3 frenaron todos en la epoca 6, lo que con patience 5 significa que su mejor val_loss fue la **epoca 1**. Las curvas del mejor modelo (`best_learning_curves.png`) lo muestran: el loss de train baja de 0.59 a 0.10, mientras el de validacion sube de ~0.52 a ~0.86 desde el inicio; la accuracy de train llega a ~0.98 mientras la de validacion se estanca en ~0.73-0.76.

Interpretacion: con TF-IDF de alta dimension, hasta una MLP moderada tiene capacidad de sobra para este dataset, y empieza a memorizar casi de inmediato. Las metricas finales son buenas porque `EarlyStopping(restore_best_weights=True)` devuelve los pesos del mejor momento de validacion, no los del final. El modelo "bueno" es, en realidad, el modelo detenido a tiempo.

## 5. La simulacion de overfitting (e4)

Configuracion pensada para fallar: 512 neuronas, solo 500 ejemplos de train (muestreo balanceado), 35 epocas y sin early stopping. Las curvas (`overfitting_learning_curves.png`) muestran el patron de manual:

- Loss de train cae practicamente a 0 hacia la epoca 5 y se queda ahi.
- Accuracy de train llega a 1.0 en ~3 epocas: memorizo los 500 ejemplos.
- Loss de validacion baja apenas al principio (~0.64) y despues sube sostenido hasta ~0.84.
- Accuracy de validacion queda estancada en ~0.64.

La brecha train-validacion (1.0 vs 0.64) es la definicion visual de sobreajuste: el modelo no aprendio el problema, aprendio el dataset. Comparado con e3 (mismo exceso de capacidad pero con datos completos y regularizacion), queda claro que el overfitting no es un accidente sino una consecuencia predecible de capacidad alta + pocos datos + entrenamiento largo sin control.

## 6. Matriz de confusion y eleccion de metrica (e3 en test, umbral 0.5)

|  | Predicho negativo | Predicho positivo |
| --- | --- | --- |
| **Real negativo** | 393 (TN) | 140 (FP) |
| **Real positivo** | 105 (FN) | 428 (TP) |

Derivados (clase positiva): precision 428/568 = 0.7535, recall 428/533 = 0.8030, F1 0.7775. Para la clase negativa, el recall (especificidad) es 393/533 = 0.7373.

Lectura del patron de errores: el modelo es levemente "optimista" — recupera el 80% de las criticas positivas, a costa de marcar como positivas 140 criticas que eran negativas (26% de los negativos). El recall positivo supera a la precision positiva en ~5 puntos: prefiere no perderse positivos aunque acepte mas falsos positivos.

Justificacion de la metrica principal: el dataset esta balanceado, asi que accuracy (0.7702) es una referencia honesta. Aun asi se elige **F1** porque resume en un solo numero el compromiso precision/recall y obliga a mirar los dos tipos de error por separado — que es exactamente lo que la matriz de confusion permite discutir. Con clases balanceadas F1 y accuracy cuentan historias parecidas (0.7775 vs 0.7702); la diferencia esta en que F1 hace visible la asimetria precision/recall que accuracy esconde.

Coherencia metodologica: val_f1 0.7647 y test_f1 0.7775 son casi iguales, senal de que la seleccion por validacion no sobreestimo el rendimiento real.

## 7. Limitaciones (para el cierre del informe)

- Umbral fijo en 0.5: no se exploro ajustarlo para mover el balance precision/recall.
- TF-IDF ignora orden y contexto de las palabras; un baseline razonable, no el techo del problema.
- Sin validacion cruzada: un solo split de validacion, mitigado por usar los splits oficiales del dataset.
- No se analizaron ejemplos mal clasificados individualmente.
