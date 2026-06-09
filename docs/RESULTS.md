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
val_accuracy: 0.7627
val_precision: 0.7583
val_recall: 0.7711
val_f1: 0.7647
test_accuracy: 0.7702
test_precision: 0.7535
test_recall: 0.8030
test_f1: 0.7775
```

## Matriz de confusion del mejor modelo

```text
                 Predicho negativo   Predicho positivo
Real negativo          393                 140
Real positivo          105                 428
```

Lectura:

- El modelo acierta 393 negativos y 428 positivos.
- Comete 140 falsos positivos.
- Comete 105 falsos negativos.
- El recall positivo es mayor que la precision positiva, por eso el modelo tiende levemente a recuperar mas positivos aunque acepte algunos falsos positivos.

## Overfitting

El archivo `results/overfitting_learning_curves.png` muestra el patron pedido por la consigna:

- Accuracy de train llega a 1.0.
- Loss de train cae casi a 0.
- Accuracy de validacion queda baja.
- Loss de validacion sube mucho.

Eso indica que el modelo memorizo el subconjunto de entrenamiento y no generalizo.
