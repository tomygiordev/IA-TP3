# `src/data.py`

## Responsabilidad

Carga el dataset desde Hugging Face y devuelve arrays de texto y etiquetas para train, validacion y test.

## Flujo

```text
load_dataset(DATASET_NAME)
        |
        v
extraer text y label
        |
        v
opcionalmente limitar train de forma balanceada
        |
        v
DatasetSplits
```

## Decision importante

El dataset ya trae splits. No se hace un split manual porque no hace falta y porque usar los splits oficiales simplifica la trazabilidad.

## Funcion de overfitting

`_limit_balanced` permite tomar una muestra chica de entrenamiento manteniendo clases balanceadas. Esto ayuda a simular overfitting sin introducir desbalance accidental.
