# Arquitectura del proyecto

Este documento describe el flujo del sistema y las decisiones metodologicas a nivel general. Los valores concretos de cada experimento (capas, neuronas, learning rate, regularizacion) viven en `src/config.py` y los resultados en `results/metrics.csv`, de modo que esta explicacion no se desactualiza con cada cambio.

## Flujo general

```text
Hugging Face Dataset
        |
        v
Carga y splits (src/data.py)
        |
        v
Vectorizacion de texto adaptada solo con train
        |
        v
MLP Keras Sequential
        |
        v
Validacion para comparar experimentos
        |
        v
Test para evaluacion final
        |
        v
Metricas, matriz de confusion y curvas
```

## Objetivo y dataset

El objetivo de la red es de clasificacion binaria de sentimiento: dada una frase de una critica de cine, predecir si expresa una opinion positiva o negativa.

Se usa `cornell-movie-review-data/rotten_tomatoes`, disponible en Hugging Face: 10.662 frases cortas de criticas de cine en ingles, con splits oficiales de 8530 ejemplos de entrenamiento, 1066 de validacion y 1066 de test. Es adecuado para el TP porque:

- Es de clasificacion binaria.
- Tiene textos cortos, por lo que entrena rapido.
- Trae splits definidos de entrenamiento, validacion y test.
- Esta balanceado, lo que permite usar accuracy como referencia sin dejar de reportar F1.

## Representacion del texto

El texto se transforma en vectores numericos con `TextVectorization` antes de las capas densas. Esto permite usar una MLP clasica sin recurrir a arquitecturas secuenciales, mantiene el problema simple y hace visible el pipeline: texto crudo, vectorizacion, capas densas, salida probabilistica.

Decision metodologica clave: el vectorizador se adapta unicamente con datos de entrenamiento. Adaptarlo con validacion o test seria data leakage.

## Modelo

La red es una MLP densa construida con `keras.Sequential`: capas ocultas con activacion no lineal y una salida sigmoide que devuelve la probabilidad de clase positiva. Sobre esa probabilidad se aplica un umbral para construir la matriz de confusion. La configuracion concreta de cada experimento esta en `src/config.py`.

## Entrenamiento y experimentos

`src/train.py` ejecuta la lista de experimentos definida en `src/config.py`. Cada experimento cambia hiperparametros controlados respecto del baseline; el detalle de que varia cada uno y por que esta en `docs/EXPERIMENTS.md`.

## Seleccion del modelo

El mejor modelo se elige por F1 de validacion, nunca por test. El test se reserva para una unica evaluacion final del modelo elegido, lo que evita contaminar la medicion de generalizacion.
