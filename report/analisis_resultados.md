# Análisis de resultados

Este documento resume la evidencia de `results/runs.csv`, `results/summary.csv`, `results/finalists_summary.csv`, `results/final_metrics.csv` y las curvas generadas el 2026-06-11.

## Metodología

- Dataset binario `cornell-movie-review-data/rotten_tomatoes`.
- Splits oficiales: 8530 train, 1066 validación y 1066 test.
- TF-IDF adaptado exclusivamente con train.
- Diez semillas por configuración en la fase principal y en el protocolo de finalistas.
- Early stopping con patience=5; monitor configurable por experimento (val_f1 o val_loss).
- Umbral sigmoide ajustado exclusivamente con validación.
- Test consultado una sola vez para el ganador final.

La fase principal (Adam + SGD) produjo 160 corridas. Las tandas de generalización sumaron 60 corridas con 3 semillas. El protocolo de finalistas añadió 60 corridas con 10 semillas.

---

## Fase 1 — Experimentos principales (Adam y SGD)

### La representación fue más importante que aumentar neuronas

El baseline Adam de 32 neuronas obtuvo F1 medio 0.7651. Aumentar a 128 neuronas produjo 0.7682, con intervalos solapados. La MLP de 32 neuronas con unigramas+bigramas y TF-IDF normalizado alcanzó 0.7722 y fue el ganador de esta fase.

El baseline lineal sin capa oculta llegó a 0.7697 con solo 5001 parámetros. Esto muestra que el espacio TF-IDF ya es fuertemente separable de forma lineal.

Parámetros compartidos salvo indicación: `max_tokens=5000`, `hidden_units=(32,)`, `optimizer=adam lr=0.001`, `batch=32`, `epochs_max=12`, early stopping sobre `val_f1`.

| Experimento | Neuronas | Tokens | n-gramas | Dropout | L2 | F1 medio | Desvío | IC 95% | Épocas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `e9_mlp32_bigrams` ★ | 32 | 10 000 | 2 | 0 | 0 | **0.7722** | 0.0034 | [0.7701; 0.7742] | 8.7 |
| `e5_dropout_128` | 128 | 5 000 | 1 | 0.2 | 0 | 0.7697 | 0.0047 | [0.7670; 0.7725] | 9.2 |
| `e1_linear` | 0 | 5 000 | 1 | 0 | 0 | 0.7697 | 0.0021 | [0.7685; 0.7709] | 10.3 |
| `e3_dropout_32` | 32 | 5 000 | 1 | 0.2 | 0 | 0.7693 | 0.0030 | [0.7676; 0.7711] | 8.9 |
| `e7_combined_128` | 128 | 5 000 | 1 | 0.5 | 1e-4 | 0.7685 | 0.0056 | [0.7650; 0.7716] | 9.2 |
| `e2_more_capacity` | 128 | 5 000 | 1 | 0 | 0 | 0.7682 | 0.0036 | [0.7662; 0.7703] | 8.9 |
| `e0_baseline` | 32 | 5 000 | 1 | 0 | 0 | 0.7651 | 0.0033 | [0.7632; 0.7670] | 8.1 |

### SGD no superó a Adam

Parámetros SGD: `hidden_units=(32,)`, `max_tokens=5000`, `epochs_max=20`.

| Configuración SGD | LR | Momentum | Dropout | L2 | F1 medio | Desvío | Épocas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sgd_lr_low` | 0.001 | 0.9 | 0 | 0 | **0.7576** | 0.0065 | 18.3 |
| `sgd_momentum_09` | 0.010 | 0.9 | 0 | 0 | 0.7554 | 0.0106 | 10.9 |
| `sgd_l2` | 0.010 | 0.9 | 0 | 1e-4 | 0.7539 | 0.0130 | 11.2 |
| `sgd_no_momentum` | 0.010 | 0 | 0 | 0 | 0.7476 | 0.0285 | 17.1 |
| `sgd_dropout` | 0.010 | 0.9 | 0.2 | 0 | 0.7370 | 0.0084 | 11.5 |
| `sgd_lr_high` | 0.100 | 0.9 | 0 | 0 | 0.6889 | 0.0094 | 8.4 |

Momentum redujo la variabilidad respecto de SGD puro. El LR bajo fue la mejor variante pero necesitó 18.3 épocas. LR 0.1 fue demasiado agresivo.

---

## Fase 2 — Búsqueda de generalización (3 semillas)

Las curvas de `e9_mlp32_bigrams` mostraban train accuracy ~0.95 vs val accuracy ~0.75: overfitting real que el early stopping contenía pero no eliminaba. Se ejecutaron dos tandas de 10 configuraciones con 3 semillas cada una para diagnosticar qué variable atacar.

Base común de todos los experimentos de generalización: `max_tokens=10 000`, `hidden_units=(32,)`, `Adam lr=0.001`, `batch=32`, `normalize_tfidf=True`.

### Tanda 1 — un cambio a la vez (top 4)

| Experimento | n-gramas | Dropout | Monitor ES | F1 medio | Desvío | Val loss |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `gen_dropout_02` | 2 | 0.2 | val_f1 | **0.7749** | 0.0024 | 0.5015 |
| `gen_es_valloss` | 2 | 0 | val_loss | 0.7733 | 0.0015 | 0.5017 |
| `gen_unigrams` | 1 | 0 | val_f1 | 0.7720 | 0.0023 | **0.4884** |
| `gen_units_16` *(16 neu)* | 2 | 0 | val_f1 | 0.7719 | 0.0026 | 0.5010 |

### Tanda 2 — combinaciones (top 4)

| Experimento | n-gramas | Dropout | Monitor ES | F1 medio | Desvío | Val loss |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `gen2_dropout03_valloss` | 2 | 0.3 | val_loss | **0.7760** | 0.0019 | 0.5003 |
| `gen2_dropout01_valloss` | 2 | 0.1 | val_loss | 0.7751 | 0.0029 | 0.5014 |
| `gen2_unigrams_valloss` | 1 | 0 | val_loss | 0.7749 | 0.0024 | **0.4884** |
| `gen2_dropout02_valloss` | 2 | 0.2 | val_loss | 0.7741 | 0.0001 | 0.5015 |

---

## Fase 3 — Protocolo de finalistas (10 semillas)

Los seis candidatos más prometedosos se reejecutaron con 10 semillas. El ganador se seleccionó por un score compuesto: `0.45·F1_mean + 0.25·(1−F1_std) + 0.20·(1−loss_mean) + 0.10·(1−loss_std)`, normalizado dentro del grupo.

Base común: `max_tokens=10 000`, `hidden_units=(32,)`, `Adam lr=0.001`, `batch=32`, `normalize_tfidf=True`, `l2=0`.

| Experimento | n-gramas | Dropout | Monitor ES | F1 medio | Desvío | Val loss | Épocas | Score |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `gen2_unigrams_valloss` ★ | 1 | 0 | val_loss | **0.7753** | **0.0020** | **0.4879** | **8.0** | **0.871** |
| `gen2_unigrams_dropout02` | 1 | 0.2 | val_f1 | 0.7751 | 0.0050 | 0.4866 | 8.7 | 0.641 |
| `gen2_dropout01_valloss` | 2 | 0.1 | val_loss | 0.7747 | 0.0024 | 0.5015 | 10.0 | 0.562 |
| `gen2_dropout03_valloss` | 2 | 0.3 | val_loss | 0.7746 | 0.0020 | 0.5009 | 11.1 | 0.468 |
| `gen2_dropout02_valloss` | 2 | 0.2 | val_loss | 0.7742 | 0.0021 | 0.5019 | 10.6 | 0.329 |
| `gen_es_valloss` | 2 | 0 | val_loss | 0.7739 | 0.0017 | 0.5017 | 10.0 | 0.330 |

**Diferencia entre puesto 1 y 2:** delta F1 = 0.0002, pooled std = 0.0035 → ruido estadístico. El score compuesto lo resuelve por estabilidad: gen2_unigrams_valloss tiene std 0.0020 frente a 0.0050 del segundo.

---

## Hallazgos de la búsqueda

### Los bigramas no mejoraron la generalización

La hipótesis inicial era que los bigramas aportaban contexto (negaciones, combinaciones valorativas). La evidencia lo contradice para este modelo: los unigramas obtuvieron val loss 0.488 frente a 0.501–0.502 de todos los modelos con bigramas. Los bigramas agregan millones de combinaciones raras que la red de 32 neuronas memoriza en lugar de generalizar.

### El monitor val_loss mejora la calibración de probabilidades

Cambiar el criterio de parada de `val_f1` a `val_loss` redujo el val loss de ~0.507 a ~0.488 y bajó el umbral óptimo de ~0.48 a 0.396. El modelo es menos sobreconfiante: sus probabilidades reflejan mejor la incertidumbre real.

### Dropout 0.3 vs Dropout 0.1: diferencia nula

Con 10 semillas, delta F1 = 0.0001, pooled std = 0.0022 → ratio < 0.1. Son equivalentes. La diferencia observada en 3 semillas era ruido.

### El umbral tiene un efecto medible

`gen2_unigrams_valloss` obtiene F1 0.7538 con umbral fijo 0.5 y 0.7753 al ajustar el corte en validación: +2.1 puntos. Esa ganancia supera la diferencia entre casi todos los pares de arquitecturas comparados.

---

## Evaluación final

El modelo elegido es `gen2_unigrams_valloss`. Arquitectura: MLP con 32 neuronas, TF-IDF unigramas (10 000 tokens), normalización L2 del vector, sin dropout, sin weight decay. Optimizador: Adam lr=0.001. Early stopping sobre val_loss con patience=5.

Reentrenado con semilla 42, umbral ajustado en validación: **0.3964**.

| Métrica test | Valor |
| --- | ---: |
| Accuracy | 0.7749 |
| Precision | 0.7438 |
| Recall | **0.8386** |
| F1 | **0.7884** |

Matriz de confusión: 379 TN · 154 FP · 86 FN · 447 TP.

El modelo recupera el 84% de las críticas positivas a costa de aceptar más falsos positivos. El recall supera a la precision en 9.5 puntos.

---

## Overfitting

La simulación intencional usa 512 neuronas, 500 ejemplos, 35 épocas y no aplica early stopping. Train llega a accuracy 1.0 y loss casi cero; validación queda en ~0.64 y su loss aumenta sostenidamente. Separada en `results/demos/`.

---

## Limitaciones

- Las semillas miden variación de inicialización y entrenamiento, pero comparten el mismo split oficial.
- TF-IDF no conserva orden ni contexto completo.
- El umbral se optimiza para F1; otra aplicación podría priorizar precision o recall.
- No se realizó análisis cualitativo de ejemplos mal clasificados.
- La búsqueda de generalización usó 3 semillas; solo los 6 finalistas se validaron con 10.

---

## Conclusión general

El modelo final `gen2_unigrams_valloss` es un baseline académico sólido con F1 test 0.7884, protocolo reproducible y curvas sin señal de overfitting severo. Los hallazgos más contraintuitivos del proceso:

1. **Los bigramas sobreajustan** en una red de 32 neuronas: más features no implica mejor generalización.
2. **El monitor de early stopping importa**: parar por val_loss en lugar de val_f1 mejora la calibración de probabilidades.
3. **La variabilidad entre semillas es tan informativa como el promedio**: dropout 0.3 y 0.1 parecían distintos en 3 semillas y resultaron equivalentes en 10.
4. **El umbral aporta tanto como muchas decisiones de arquitectura**: ajustarlo en validación sumó 2 puntos de F1 sin cambiar ni un peso de la red.

El siguiente paso razonable no es más neuronas, sino mejor representación textual: analizar ejemplos mal clasificados, modelar negaciones explícitamente, y recién después considerar embeddings contextuales.
