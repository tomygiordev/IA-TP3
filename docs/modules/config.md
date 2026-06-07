# `src/config.py`

## Responsabilidad

Centraliza parametros del proyecto:

- Nombre del dataset.
- Semilla de reproducibilidad.
- Carpeta de resultados.
- Umbral de clasificacion.
- Lista de experimentos.

## Por que existe

Evita que hiperparametros importantes queden dispersos entre varios archivos. Si el grupo quiere agregar un experimento, debe hacerlo aca.

## Puntos importantes

- `ExperimentConfig` usa `dataclass(frozen=True)` para que cada configuracion sea inmutable.
- `EXPERIMENTS` define todos los casos que corre `train.py`.
- `e4_overfitting_demo` usa `train_limit=500` para forzar sobreajuste.
