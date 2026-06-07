# `src/train.py`

## Responsabilidad

Orquesta el entrenamiento completo.

## Flujo

```text
leer experimento
cargar datos
adaptar vectorizador con train
construir modelo
entrenar
evaluar validacion y test
guardar metricas
elegir mejor por val_f1
guardar graficos
```

## Seleccion del mejor modelo

El mejor se elige con `val_f1`. Esto evita mirar test para tomar decisiones de ajuste.

## Early stopping

Los experimentos normales usan `EarlyStopping` con `restore_best_weights=True`.

El experimento de overfitting no usa early stopping, porque su objetivo es mostrar explicitamente como se separan train y validacion.
