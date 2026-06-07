# TP 3 IA - MLP para clasificacion binaria

Proyecto base para el TP 3 de Inteligencia Artificial sobre redes neuronales multicapa. El repositorio entrena una MLP con Keras Sequential sobre un dataset binario de Hugging Face, ejecuta experimentos de hiperparametros, simula overfitting y genera evidencia para defender el informe.

El foco no es hacer el modelo mas grande posible, sino construir una base academica correcta, reproducible y facil de explicar.

## Estado del proyecto

- Dataset cargado desde Hugging Face: `cornell-movie-review-data/rotten_tomatoes`.
- Tarea: clasificacion binaria de sentimiento.
- Modelo: MLP densa con `tf.keras.Sequential`.
- Entrada: texto vectorizado con `TextVectorization` en modo `tf_idf`.
- Evaluacion: matriz de confusion, accuracy, precision, recall y F1.
- Overfitting: simulado con pocos datos y una arquitectura sobredimensionada.
- Resultados: generados en `results/`.

## Por que esta implementacion es defendible

La consigna pide Keras Sequential, clasificacion binaria, dataset de Hugging Face, MLP, overfitting, experimentos de hiperparametros y matriz de confusion. Este proyecto cubre esos puntos sin agregar arquitecturas que confundan la defensa, como Transformers, RNN o CNN.

Las decisiones siguen recomendaciones de fuentes profesionales:

- Keras recomienda `Sequential` para stacks lineales de capas con una entrada y una salida por capa.
- TensorFlow recomienda adaptar `TextVectorization` solo con datos de entrenamiento para evitar filtracion de informacion.
- AWS recomienda separar entrenamiento, validacion y test para evitar data leakage y usar validacion para elegir hiperparametros.
- Google ML Crash Course explica que accuracy es util en datasets balanceados, pero precision, recall y F1 permiten discutir mejor los tipos de error.

## Estructura

```text
.
|-- README.md
|-- requirements.txt
|-- AGENTS.md
|-- src/
|   |-- config.py
|   |-- data.py
|   |-- model.py
|   |-- evaluate.py
|   |-- train.py
|   |-- smoke_test.py
|   `-- __init__.py
|-- results/
|   |-- metrics.csv
|   |-- confusion_matrix.png
|   |-- best_learning_curves.png
|   |-- learning_curves.png
|   `-- overfitting_learning_curves.png
|-- docs/
|   |-- repo_explorer.html
|   |-- FILE_MAP.md
|   |-- ARCHITECTURE.md
|   |-- EXPERIMENTS.md
|   |-- RESULTS.md
|   |-- SOURCES.md
|   |-- REVIEWER_NOTES.md
|   |-- materials/
|   `-- modules/
|-- .agents/
|   `-- commit-policy.md
|-- report/
|   `-- guia_informe.md
|-- notebooks/
`-- data/
```

## Instalacion

Se recomienda usar Python 3.10 para TensorFlow.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si pip falla por certificados SSL en Windows, puede usarse:

```powershell
python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## Ejecucion rapida

Prueba minima para verificar entorno, dataset, vectorizador, modelo y metricas:

```powershell
python src\smoke_test.py
```

Entrenamiento completo de los experimentos:

```powershell
python src\train.py
```

## Resultados actuales

El mejor experimento por F1 de validacion fue `e3_regularized`, con Dropout y L2:

```text
val_f1:         0.7577
test_accuracy:  0.7580
test_precision: 0.7442
test_recall:    0.7861
test_f1:        0.7646
```

La matriz de confusion final esta en:

```text
results/confusion_matrix.png
```

La simulacion de overfitting esta en:

```text
results/overfitting_learning_curves.png
```

## Explorador interactivo

Abrir en el navegador:

```text
docs/repo_explorer.html
```

Ese HTML explica de forma visual el flujo del repositorio, la arquitectura MLP, los experimentos y los argumentos para defender cada decision ante una revision critica.

## Documentacion

- [Mapa de archivos](docs/FILE_MAP.md)
- [Arquitectura](docs/ARCHITECTURE.md)
- [Experimentos](docs/EXPERIMENTS.md)
- [Resultados](docs/RESULTS.md)
- [Fuentes profesionales](docs/SOURCES.md)
- [Notas para revision critica](docs/REVIEWER_NOTES.md)
- [Guia del informe](report/guia_informe.md)

## Instrucciones para agentes

La regla general para cualquier agente está en `AGENTS.md`.

Además, el repositorio incluye una guía complementaria en `.agents/commit-policy.md` para detallar cómo deben redactarse los commits por feature.

