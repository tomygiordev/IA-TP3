# Notas para una revision critica

Este documento anticipa observaciones posibles y deja respuestas tecnicas preparadas.

## "Por que no usaron Transformers?"

Porque la consigna pide MLP y Keras Sequential. Usar Transformers resolveria otra consigna y haria mas dificil defender que se estudio una red neuronal multicapa densa.

## "Por que texto si una MLP no entiende secuencia?"

La MLP no recibe texto secuencial. Recibe una representacion vectorial `tf_idf`. Esto es deliberado: permite trabajar con capas densas y mantener la arquitectura dentro de la consigna.

## "Por que F1 y no solo accuracy?"

El dataset es balanceado, por lo que accuracy sirve como referencia. Aun asi, F1 resume precision y recall, y permite discutir falsos positivos y falsos negativos con mas criterio.

## "Por que el mejor modelo no es el de mayor test_f1?"

Porque el modelo se selecciona con validacion, no con test. El test se reserva para evaluacion final. Elegir por test seria una forma de contaminar la evaluacion.

## "Por que el modelo sobreajusta tan rapido?"

El texto vectorizado con `tf_idf` genera una entrada de alta dimension. Si la red tiene mucha capacidad, puede memorizar patrones del entrenamiento rapidamente. Eso es util para demostrar overfitting.

## "Que limitaciones tiene?"

- No usa embeddings entrenables ni modelos preentrenados.
- No explora ajuste de umbral.
- No hace validacion cruzada.
- No analiza ejemplos mal clasificados.

Estas limitaciones son aceptables para una base academica inicial y pueden proponerse como trabajo futuro.
