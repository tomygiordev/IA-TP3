# Guia para redactar el informe

La consigna permite IA para dudas, codigo y graficos, pero prohibe usar IA para escribir el informe. Este archivo es solo una estructura de trabajo para que el grupo lo redacte con sus propias palabras.

## Pagina 1 - Problema y dataset

- Explicar que se aborda clasificacion binaria de sentimiento.
- Indicar dataset de Hugging Face: `cornell-movie-review-data/rotten_tomatoes`.
- Describir columnas: texto de critica y etiqueta binaria.
- Justificar que es adecuado para arrancar: pequeno, binario, con splits definidos.

## Pagina 2 - Modelo y preparacion

- Explicar TextVectorization en modo `tf_idf`.
- Aclarar que el vectorizador se adapta solo con train.
- Describir MLP: capas densas, ReLU, salida sigmoide, binary crossentropy.

## Pagina 3 - Overfitting

- Mostrar curvas del experimento de overfitting.
- Explicar diferencia entre error de entrenamiento y validacion.
- Relacionarlo con capacidad excesiva y pocos datos.

## Pagina 4 - Experimentos

- Incluir tabla con al menos 4 variaciones de hiperparametros.
- Comparar arquitectura, learning rate, dropout/L2, batch size o capacidad.
- Explicar que se eligio el mejor por F1 de validacion.

## Pagina 5 - Evaluacion final

- Mostrar matriz de confusion de test.
- Reportar accuracy, precision, recall y F1.
- Defender metrica principal elegida.
- Cerrar con limitaciones y posible mejora futura.
