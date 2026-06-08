# Prompts de evaluacion y defensa

## Prompt 1 - elegir la metrica principal con criterio

```text
mi dataset binario parece bastante balanceado, pero igual tengo que reportar y explicar accuracy, precision, recall y f1, y ademas elegir cual voy a usar como referencia principal para comparar experimentos y seleccionar el mejor modelo.

quiero que me ayudes a pensar esa decision con criterio academico de verdad. no me des una respuesta automatica del estilo "usa f1 porque si", porque despues si me preguntan por que la elegi y yo solo repito eso, queda flojo. necesito entender cuando accuracy alcanza como referencia razonable, cuando conviene mirar precision y recall por separado, y por que podria tener sentido ordenar los experimentos por f1 de validacion aun cuando el dataset no este desbalanceado de forma extrema.

tambien quiero que me ayudes a construir una justificacion equilibrada: algo que reconozca que accuracy en un dataset balanceado no esta mal, pero que al mismo tiempo explique por que f1 puede ser una metrica util para resumir el compromiso entre precision y recall y evitar una lectura demasiado superficial del rendimiento.

si podes, inclui ejemplos de situaciones donde accuracy sola esconderia cosas importantes y donde mirar precision o recall por separado cambia la interpretacion del comportamiento del modelo.
```

## Prompt 2 - leer la matriz de confusion sin decir obviedades

```text
tengo una matriz de confusion de un modelo binario y no quiero quedarme en una lectura plana de "aca hay tantos verdaderos positivos, aca tantos falsos negativos". ayudame a construir una interpretacion mas inteligente y mas conectada con el comportamiento real del modelo.

me interesa que la explicacion relacione la matriz con precision, recall y el tipo de errores que esta cometiendo el sistema. por ejemplo, quiero poder detectar si el modelo esta priorizando recuperar positivos aunque eso implique aceptar mas falsos positivos, o si al reves esta siendo conservador y pierde recall para no equivocarse tanto al marcar positivos.

tambien necesito una forma de contarlo que suene a que de verdad entiendo lo que estoy viendo. no quiero una explicacion mecanica. quiero algo que me sirva tanto para una defensa oral como para una seccion corta del informe, donde se vea que no solo se calcularon metricas sino que se interpreto el patron de errores del modelo.

si podes, sumame alguna recomendacion sobre que detalles vale la pena remarcar y cuales son obvios o redundantes cuando uno presenta una matriz de confusion en un contexto academico.
```

## Prompt 3 - pedir una revision critica del tp completo

```text
quiero que revises de forma exigente un tp de clasificacion binaria hecho con una mlp en keras sequential. el proyecto usa un dataset de hugging face, arranca con una baseline chica, despues prueba variantes con mas capacidad, cambio de learning rate, regularizacion y una simulacion de overfitting. ademas reporta accuracy, precision, recall, f1 y matriz de confusion.

necesito una evaluacion honesta sobre si esto se ve aprobable en un contexto universitario o si tiene agujeros fuertes. no quiero elogios de compromiso ni una respuesta complaciente. quiero que me señales con claridad que cosas estan bien planteadas, cuales son defendibles pero algo flojas, y cuales realmente podrian hacer que el trabajo pierda puntos frente a una correccion critica.

si ves decisiones razonables, explicame por que lo son y bajo que criterio las defenderias. si ves humo, simplificaciones injustificadas o partes donde el proyecto parece mas prolijo de lo que realmente demuestra, marcame exactamente donde estaria el problema. me sirve mas una devolucion dura pero util que una aprobacion tibia.

tambien me interesaria que, si encontrás puntos flojos, me digas si son fallas graves de metodologia o si son mas bien omisiones corregibles para una entrega de este nivel. la idea es usar esa revision para cerrar mejor el trabajo antes de entregarlo.
```
