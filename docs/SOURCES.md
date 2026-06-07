# Fuentes profesionales usadas

Estas fuentes respaldan las decisiones tecnicas del proyecto. Se priorizaron portales oficiales o documentacion profesional.

## Keras Sequential

Fuente: https://keras.io/guides/sequential_model/

Uso en el proyecto:

- Justifica usar `keras.Sequential` para un stack lineal de capas.
- Refuerza que conviene especificar forma de entrada cuando se conoce.

## TensorFlow Text Classification

Fuente: https://www.tensorflow.org/tutorials/keras/text_classification

Uso en el proyecto:

- Respalda el uso de `TextVectorization`.
- Advierte que `adapt` debe hacerse solo con entrenamiento para no filtrar informacion del test.

## TensorFlow Overfit and Underfit

Fuente: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit

Uso en el proyecto:

- Respalda comparar modelos chicos y grandes.
- Respalda usar L2 y Dropout como tecnicas de regularizacion.

## Hugging Face Datasets

Fuente: https://huggingface.co/docs/datasets/loading

Uso en el proyecto:

- Respalda cargar datasets desde el Hub con `load_dataset`.
- Permite trabajar con splits definidos.

## AWS Prescriptive Guidance: Splits and Data Leakage

Fuente: https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-operations-planning/splits-leakage.html

Uso en el proyecto:

- Respalda separar entrenamiento, validacion y test.
- Respalda usar validacion para elegir hiperparametros.
- Respalda reservar test para evaluacion final.

## Google ML Crash Course: metricas

Fuente: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

Uso en el proyecto:

- Respalda reportar accuracy, precision, recall y F1.
- Explica que accuracy es razonable como referencia en datasets balanceados.

## Google ML Crash Course: threshold y matriz de confusion

Fuente: https://developers.google.com/machine-learning/crash-course/classification/thresholding

Uso en el proyecto:

- Respalda convertir probabilidades en clases mediante un umbral.
- Respalda analizar falsos positivos y falsos negativos con matriz de confusion.
