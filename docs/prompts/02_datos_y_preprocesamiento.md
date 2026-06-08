# Prompts de datos y preprocesamiento

## Prompt 1 - vectorizar texto sin salirme de la consigna

```text
voy a resolver un problema de clasificacion binaria de texto con una mlp, asi que no voy a modelar secuencias con una red recurrente ni con transformers. por eso necesito una forma sensata de transformar el texto en una entrada numerica que sea compatible con capas densas y que, al mismo tiempo, siga siendo facil de explicar en un tp.

quiero que me expliques si tiene sentido usar TextVectorization de keras en modo tf_idf antes de las capas densas. me interesa entenderlo bien, no solo usarlo porque "funciona". necesito saber que informacion conserva esa representacion, que pierde respecto de una representacion secuencial, por que puede ser una buena decision cuando la consigna me pide trabajar con una mlp, y que ventajas practicas tiene desde el punto de vista de entrenamiento y simplicidad del pipeline.

tambien me gustaria que me marques las limitaciones de este enfoque. por ejemplo, hasta que punto una representacion tf-idf ignora orden y contexto, y por que aun asi puede servir para una clasificacion de sentimiento binaria como baseline academica. si podes, comparalo brevemente contra otras alternativas mas complejas, pero dejando claro por que tf-idf + mlp podria ser la opcion mas coherente para este caso.

ademas, marcame errores comunes o puntos delicados del preprocesamiento. especialmente me interesa cualquier cosa que pueda generar data leakage entre train, validation y test, como adaptar el vectorizador con splits que no corresponden, o decisiones que despues puedan contaminar la evaluacion final sin que uno se de cuenta.
```

## Prompt 2 - revisar el split y el flujo de datos

```text
te paso la idea general de mi pipeline y quiero que la revises con mirada critica, como si estuvieras corrigiendo metodologia y no solo viendo si el codigo "anda". el flujo que tengo en mente es este: cargo un dataset binario desde hugging face, uso los splits que ya trae, adapto el vectorizador solo con train, entreno los modelos con train, comparo hiperparametros usando validation y dejo test reservado unicamente para la evaluacion final del mejor modelo.

decime si este esquema esta bien planteado desde el punto de vista metodologico y si hay algo que te haria ruido. me interesa en particular saber si la logica de separacion entre train, validation y test esta bien usada, si hay riesgos de fuga de informacion aunque no sean obvios, y si la manera de elegir el "mejor modelo" podria inducir una conclusion injusta o sesgada.

tambien quiero que me digas que cosas deberia poder explicar si el profesor me pregunta por que use validation para comparar experimentos y no test. a veces uno hace estas cosas por costumbre, pero yo necesito poder argumentarlas bien. si hay posibles mejoras o controles extra que harian el flujo mas solido, mencionamelos, aunque despues me quede con la version mas simple por una cuestion de alcance del tp.

prefiero una devolucion sincera antes que una aprobacion automatica. si ves algo flojo, decimelo sin problema y explicame por que.
```

## Prompt 3 - justificar una red inicial chica

```text
el profesor me sugirio arrancar con una red neuronal chica e ir puliendola o agrandandola segun haga falta. en ese contexto, yo quiero partir de una mlp bastante simple para texto vectorizado, por ejemplo una sola capa oculta con pocas neuronas, y usar eso como baseline antes de meter mas capacidad o regularizacion.

necesito que me ayudes a justificar academicamente por que esa forma de arrancar tiene sentido. no quiero una explicacion vacia de manual, sino una argumentacion que conecte varias ideas: capacidad del modelo, riesgo de sobreajuste, facilidad de interpretacion, costo de entrenamiento, claridad para comparar experimentos y utilidad del baseline como punto de referencia.

me serviria que lo plantees tambien en contraste con la mala decision de arrancar directamente con una red grande "porque total despues veo". quiero poder explicar por que eso complica la lectura de resultados: si un modelo grande sale mas o menos bien desde el principio, despues es mas dificil saber si de verdad hacia falta tanta capacidad o si una red mucho mas simple ya resolvia el problema de manera aceptable.

si podes, ligalo con el caso concreto de una mlp sobre texto vectorizado con tf-idf, donde la dimensionalidad de entrada puede ser alta y por eso una red con demasiados parametros puede memorizar patrones demasiado especificos del train.
```
