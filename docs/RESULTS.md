# Resultados

La corrida completa se genera con:

```bash
python src/train.py
```

Para ejecutar únicamente las variantes SGD pendientes, sin repetir las corridas ya guardadas ni la simulación de overfitting:

```bash
python src/train.py --sgd-only
```

## Archivos principales

| Archivo | Contenido |
| --- | --- |
| `results/runs.csv` | Una fila por configuración y semilla. |
| `results/summary.csv` | Media, desvío e intervalo bootstrap de validación. |
| `results/final_metrics.csv` | Única evaluación del modelo seleccionado sobre test. |
| `results/confusion_matrix.png` | Matriz de confusión final. |
| `results/best_learning_curves.png` | Curvas del modelo ganador. |
| `results/overfitting_learning_curves.png` | Simulación intencional de sobreajuste. |

## Protocolo

- 16 configuraciones: 10 iniciales con Adam y 6 variantes SGD.
- 10 semillas por configuración: 160 corridas únicas.
- Selección por F1 medio de validación con umbral ajustado en validación.
- Test reservado para el ganador global.
- Early stopping sobre F1 de validación, salvo en la simulación de overfitting.

## Mejor modelo

`e9_mlp32_bigrams`: Adam, 32 neuronas, unigramas+bigrams y normalización L2 del vector TF-IDF.

```text
F1 validación ajustado medio: 0.7722 ± 0.0034
IC bootstrap 95%: [0.7701, 0.7742]
Umbral final elegido en validación: 0.4851
Accuracy test: 0.7824
Precision test: 0.7573
Recall test: 0.8311
F1 test: 0.7925
```

Matriz de confusión:

```text
                 Predicho negativo   Predicho positivo
Real negativo          391                 142
Real positivo           90                 443
```

## Resultado de SGD

La mejor variante fue `sgd_lr_low`, con LR 0.001 y momentum 0.9:

```text
F1 validación ajustado medio: 0.7576 ± 0.0065
IC bootstrap 95%: [0.7536, 0.7612]
```

SGD no superó a Adam. Sin momentum aumentó mucho la variabilidad; LR 0.1 degradó el resultado y dropout 0.2 produjo subajuste.

## Overfitting

La simulación usa 512 neuronas, 500 ejemplos, 35 épocas y ningún early stopping. Train llega a accuracy 1.0 y loss casi cero, mientras validación queda cerca de 0.64 y su loss crece. Sus resultados se interpretan por separado y no participan en la selección del modelo.
