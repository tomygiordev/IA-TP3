# Arquitectura del proyecto

## Flujo general

```text
Hugging Face Dataset
        |
        v
src/data.py
        |
        v
TextVectorization adaptado solo con train
        |
        v
MLP Keras Sequential
        |
        v
Validacion para elegir hiperparametros
        |
        v
Test para evaluacion final
        |
        v
Metricas, matriz de confusion y curvas
```

## Dataset

Se usa `cornell-movie-review-data/rotten_tomatoes`, disponible en Hugging Face. Es adecuado para el TP porque:

- Es de clasificacion binaria.
- Tiene textos cortos, por lo que entrena rapido.
- Tiene splits definidos de entrenamiento, validacion y test.
- Esta balanceado, lo que permite usar accuracy como referencia sin dejar de reportar F1.

## Representacion del texto

El texto se transforma con `TextVectorization` en modo `tf_idf`.

Decision defendible:

- Permite usar una MLP densa sin recurrir a Transformers, RNN ni CNN.
- Mantiene el problema simple.
- Hace visible el pipeline clasico: texto crudo, vectorizacion, capas densas, salida probabilistica.

Punto critico:

- El vectorizador se adapta solo con datos de entrenamiento. Adaptarlo con validacion o test seria data leakage.

## Modelo

La arquitectura base es:

```text
Input texto
TextVectorization tf_idf
Dense ReLU
Dense sigmoid
```

La salida `sigmoid` devuelve una probabilidad de clase positiva. Luego se aplica umbral `0.5` para construir la matriz de confusion.

## Entrenamiento

`train.py` ejecuta una lista de experimentos definidos en `config.py`. Cada experimento cambia hiperparametros controlados:

- Cantidad de neuronas.
- Profundidad.
- Learning rate.
- Dropout.
- Regularizacion L2.
- Cantidad de datos para simular overfitting.

## Seleccion del modelo

El mejor modelo se elige por `val_f1`, no por test. Esto evita usar el conjunto de test como criterio de ajuste.

Despues de elegir el mejor por validacion, se reportan metricas finales en test.
