# `src/evaluate.py`

## Responsabilidad

Contiene la evaluacion del modelo:

- Convierte probabilidades en clases.
- Calcula accuracy, precision, recall y F1.
- Guarda matriz de confusion.
- Guarda curvas de aprendizaje.

## Umbral

Se usa `THRESHOLD = 0.5`, definido en `config.py`.

Esto debe aclararse porque la matriz de confusion depende del umbral elegido.

## Metricas

Se calculan con scikit-learn para tener funciones estandarizadas y faciles de auditar.
