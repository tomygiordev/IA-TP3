# Prompts de experimentos y overfitting

## Prompt 1 - armar una bateria de experimentos con criterio

```text
ya tengo una baseline chica para clasificacion binaria con una mlp en keras sequential y ahora necesito planear experimentos de hiperparametros. la consigna pide al menos 4, pero no quiero hacer variaciones al azar solo para llenar una tabla o aparentar exploracion donde en realidad no hay una hipotesis detras.

quiero que me propongas una secuencia corta pero consistente de experimentos que realmente tenga sentido comparar. por ejemplo: una baseline simple, despues un modelo con mas capacidad, otro con learning rate distinto, otro con alguna forma de regularizacion como dropout o l2, y por separado algun caso pensado para mostrar overfitting de forma bien visible. si se te ocurren variantes mejores, decimelas, pero justificando el por que.

para cada experimento necesito que me aclares varias cosas: que hiperparametro o decision cambia respecto del anterior, que hipotesis estaria poniendo a prueba, que esperaria ver si esa intuicion fuera correcta, y como interpretaria un resultado contrario. me interesa que no quede como una lista decorativa, sino como una mini estrategia de exploracion.

tambien me gustaria que me señales que errores comunes hay al comparar experimentos. por ejemplo, cambiar demasiadas cosas juntas y despues no saber que produjo la mejora o el empeoramiento. la idea es que las comparaciones me sirvan para aprender algo del comportamiento del modelo, no solo para generar una tabla con numeritos.
```

## Prompt 2 - provocar overfitting sin falsear el trabajo

```text
necesito simular overfitting de manera intencional para un tp, pero quiero hacerlo de una forma honesta y explicable. no busco "trucar" el trabajo ni fabricar un resultado artificial sin sentido, sino armar un escenario donde se vea con claridad que el modelo memoriza el conjunto de entrenamiento y despues generaliza mal sobre validation o test.

decime una estrategia razonable para lograrlo en una mlp aplicada a texto vectorizado. quiero que pienses en variables concretas como capacidad del modelo, cantidad de ejemplos de entrenamiento, numero de epocas y ausencia de regularizacion. me interesa saber que combinacion de esas cosas aumenta la probabilidad de que aparezca sobreajuste de forma evidente y defendible.

ademas, explicame que señales deberia mirar en las curvas de loss y accuracy para poder decir con fundamento "aca hubo overfitting". no quiero quedarme solo con una frase de manual. quiero saber, por ejemplo, que significa ver accuracy de train muy alta mientras validation se estanca, o loss de train bajando casi a cero mientras la de validation sube fuerte. si podes, sumame una explicacion intuitiva de por que pasa eso en una red con demasiada capacidad para pocos datos.

tambien me sirve que me digas como presentar este experimento sin que parezca un error del proyecto, sino una demostracion intencional de un fenomeno que justamente era parte de la consigna.
```

## Prompt 3 - interpretar por que una red mas grande no siempre gana

```text
si en mis experimentos una red con mas capas o mas neuronas entrena mejor pero no mejora en validacion, quiero poder explicarlo bien y sin frases vacias del tipo "mas grande no siempre es mejor". ayudame a construir una interpretacion tecnica clara de por que mayor capacidad no implica automaticamente mejor generalizacion.

quiero que la explicacion se apoye en el problema concreto que estoy resolviendo: texto vectorizado con tf-idf y una mlp densa. me interesa que se entienda que, al trabajar con una entrada de alta dimensionalidad, una red con muchos parametros puede aprender patrones muy finos del conjunto de entrenamiento, incluso patrones poco robustos o demasiado ligados a palabras y combinaciones que no generalizan bien.

ademas, explicame como enlazarias esa observacion con un experimento posterior de regularizacion. o sea, si una red mas capaz empieza a sobreajustar, como justifico que probar dropout o regularizacion l2 no es "tirar magia" sino una respuesta razonable a lo que mostraron las curvas y las metricas de validacion.

si podes, dame una version mas conceptual y otra mas corta, como para poder usar una en el informe y otra en una defensa oral si me lo preguntan en vivo.
```
