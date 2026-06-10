# Experimentos

La consigna pide variar hiperparametros en al menos 4 experimentos y explicar resultados. Este proyecto define 5 casos para cubrir comparacion, regularizacion y overfitting. Cada experimento cambia una sola cosa respecto del baseline, para poder atribuirle el resultado a esa variable. Los valores exactos estan en `src/config.py`.

## Tabla de experimentos

| Experimento | Que cambia respecto del baseline | Hipotesis que prueba |
| --- | --- | --- |
| `e0_baseline` | Nada: MLP chica de referencia | Una red simple ya resuelve el problema de forma aceptable. |
| `e1_more_capacity` | Mas neuronas en la capa oculta | Mas capacidad puede ajustar mejor, pero no garantiza generalizar mejor. |
| `e2_lower_learning_rate` | Learning rate mas bajo | Un entrenamiento mas lento y estable no necesariamente gana en validacion. |
| `e3_regularized` | Mas capacidad + dropout + L2 | La regularizacion controla el sobreajuste del modelo con mas capacidad. |
| `e4_overfitting_demo` | Pocos datos, mucha capacidad, mas epocas y sin early stopping | Una red sobredimensionada memoriza el train y generaliza mal. |

## Criterio de comparacion

Se ordena por F1 de validacion (`val_f1`). F1 combina precision y recall, y validacion es el conjunto correcto para comparar hiperparametros: el test queda reservado para la evaluacion final.

Los resultados numericos vigentes estan en [RESULTS.md](RESULTS.md) y en `results/metrics.csv`.

## Interpretacion breve

- El baseline funciona bien, pero sobreajusta despues de pocas epocas.
- Aumentar neuronas mejora el ajuste en train rapido, pero no garantiza mejor generalizacion.
- Bajar el learning rate entrena mas suave, aunque no necesariamente gana en validacion.
- Dropout + L2 ayudan a controlar el exceso de capacidad.
- El experimento de overfitting memoriza entrenamiento y falla en validacion, que es exactamente lo que se queria demostrar.
