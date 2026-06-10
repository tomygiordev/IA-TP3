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
|-- README.md          -> presentacion del proyecto
|-- requirements.txt   -> dependencias
|-- AGENTS.md          -> instrucciones para agentes, incluida la politica de commits
|-- src/               -> codigo: configuracion, datos, modelo, entrenamiento y evaluacion
|-- results/           -> evidencia generada: metricas y graficos
|-- docs/              -> documentacion tecnica y explicativa
|   |-- materials/     -> consigna y teoria de catedra
|   `-- prompts/       -> prompts usados como apoyo durante el TP
`-- report/            -> guia para redactar el informe final
```

## Instalacion

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecucion rapida

Prueba minima para verificar entorno, dataset, vectorizador, modelo y metricas:

```bash
python src/smoke_test.py
```

Entrenamiento completo de los experimentos:

```bash
python src/train.py
```

## Resultados

El mejor experimento se selecciona por F1 de validacion y se evalua una sola vez en test. Los numeros vigentes estan en [docs/RESULTS.md](docs/RESULTS.md) y en `results/metrics.csv`. La matriz de confusion final esta en `results/confusion_matrix.png` y la simulacion de overfitting en `results/overfitting_learning_curves.png`.

## Explorador interactivo

Abrir en el navegador:

```text
docs/repo_explorer.html
```

Ese HTML explica de forma visual el flujo del repositorio, la arquitectura MLP, los experimentos y el porque de cada decision tecnica.

## Documentacion

- [Arquitectura](docs/ARCHITECTURE.md)
- [Experimentos](docs/EXPERIMENTS.md)
- [Resultados](docs/RESULTS.md)
- [Fuentes profesionales](docs/SOURCES.md)
- [Notas para revision critica](docs/REVIEWER_NOTES.md)
- [Apartado de prompts](docs/prompts/README.md)
- [Guia del informe](report/guia_informe.md)

## Instrucciones para agentes

La regla general para cualquier agente, incluida la politica de commits por feature, está en `AGENTS.md`.

