# Experimentos

La consigna pide variar hiperparametros en al menos 4 experimentos y explicar resultados. Este proyecto define 5 casos para cubrir comparacion, regularizacion y overfitting.

## Tabla de experimentos

| Experimento | Cambio principal | Objetivo |
| --- | --- | --- |
| `e0_baseline` | MLP chica con una capa de 32 neuronas | Tener una referencia simple. |
| `e1_more_capacity` | Dos capas: 128 y 64 neuronas | Ver si mas capacidad mejora el ajuste. |
| `e2_lower_learning_rate` | Learning rate `0.0001` | Observar entrenamiento mas lento/estable. |
| `e3_regularized` | Dropout `0.5` + L2 `0.0001` | Reducir sobreajuste. |
| `e4_overfitting_demo` | 500 ejemplos y red grande | Simular overfitting de forma intencional. |

## Criterio de comparacion

Se ordena por F1 de validacion (`val_f1`). La razon es que F1 combina precision y recall, y validacion es el conjunto correcto para comparar hiperparametros.

## Resultado actual

El mejor por `val_f1` fue:

```text
e3_regularized
hidden_units: (128, 64)
dropout: 0.5
l2: 0.0001
val_f1: 0.7577
test_f1: 0.7646
```

## Interpretacion breve

- El baseline funciona bien, pero sobreajusta despues de pocas epocas.
- Aumentar capacidad mejora train rapido, pero no garantiza mejor generalizacion.
- Bajar learning rate entrena mas suave, aunque no necesariamente gana en validacion.
- Dropout + L2 ayuda a controlar el exceso de capacidad.
- El experimento de overfitting memoriza entrenamiento y falla en validacion, que es exactamente lo que se queria demostrar.
