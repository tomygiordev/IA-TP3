# Resultados

Los resultados se generan con:

```powershell
python src\train.py
```

## Archivos generados

| Archivo | Contenido |
| --- | --- |
| `results/metrics.csv` | Tabla completa de experimentos y metricas. |
| `results/confusion_matrix.png` | Matriz de confusion del mejor modelo en test. |
| `results/best_learning_curves.png` | Curvas del mejor modelo. |
| `results/overfitting_learning_curves.png` | Curvas del experimento de overfitting. |

## Mejor modelo actual

```text
experiment: e3_regularized
val_accuracy: 0.7570
val_precision: 0.7556
val_recall: 0.7598
val_f1: 0.7577
test_accuracy: 0.7580
test_precision: 0.7442
test_recall: 0.7861
test_f1: 0.7646
```

## Matriz de confusion del mejor modelo

```text
                 Predicho negativo   Predicho positivo
Real negativo          389                 144
Real positivo          114                 419
```

Lectura:

- El modelo acierta 389 negativos y 419 positivos.
- Comete 144 falsos positivos.
- Comete 114 falsos negativos.
- El recall positivo es mayor que la precision positiva, por eso el modelo tiende levemente a recuperar mas positivos aunque acepte algunos falsos positivos.

## Overfitting

El archivo `results/overfitting_learning_curves.png` muestra el patron pedido por la consigna:

- Accuracy de train llega a 1.0.
- Loss de train cae casi a 0.
- Accuracy de validacion queda baja.
- Loss de validacion sube mucho.

Eso indica que el modelo memorizo el subconjunto de entrenamiento y no generalizo.
