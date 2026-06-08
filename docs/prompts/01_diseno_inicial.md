# Prompts de diseño inicial

## Prompt 1 - bajar la consigna a una solucion concreta

```text
estoy haciendo un tp de inteligencia artificial sobre redes neuronales multicapa, y quiero bajar la consigna a una implementacion concreta sin irme a algo mas complejo de lo que me estan pidiendo. la consigna dice basicamente esto: usar keras con modelo sequential, elegir un dataset de hugging face que sea de clasificacion binaria, entrenar una red neuronal multicapa para resolver el problema, simular overfitting, hacer al menos 4 experimentos variando hiperparametros y despues explicar los resultados con matriz de confusion y una metrica adecuada.

mi idea no es impresionar con una arquitectura rara, sino hacer algo que este bien planteado y que despues pueda defender oralmente sin vender humo. por eso quiero evitar transformers, rnns o cnns, salvo que hubiera una razon realmente muy fuerte, porque siento que me correria del espiritu del trabajo. me interesa una mlp clasica, clara, chica de entrada, y despues ir viendo si conviene ampliarla o regularizarla segun lo que muestren los resultados.

quiero que me propongas una solucion completa pero razonable, pensada para un tp universitario: tipo de dataset que conviene elegir, como representar la entrada si el dato fuera texto, como seria una arquitectura base chica en keras sequential, que funcion de activacion pondrias en las capas ocultas y en la salida, que loss usarias, que optimizador seria una buena opcion para arrancar, y como separarias train, validation y test sin mezclar roles.

tambien necesito que me expliques el criterio detras de cada decision. no quiero una receta cerrada, quiero entender por que esa propuesta es una buena forma de arrancar. por ejemplo, por que conviene tener un baseline simple antes de meter mas capacidad, por que reservar validation para comparar experimentos y dejar test solo para evaluacion final, y por que esa estrategia me ayudaria despues a mostrar overfitting de forma clara.

si ves varias opciones razonables, comparamelas un poco, pero quedate con una recomendacion principal y decime por que esa seria la mejor para este contexto. prioriza claridad conceptual, facilidad de implementacion, coherencia con la consigna y posibilidad de justificar todo despues en un informe corto.
```

## Prompt 2 - decidir si el dataset elegido tiene sentido

```text
tengo que elegir un dataset de hugging face para un problema de clasificacion binaria y resolverlo con una mlp en keras sequential. estoy evaluando usar rotten tomatoes porque, a primera vista, parece una opcion bastante manejable: el problema es binario, el dataset no parece imposible de entrenar en una pc comun, y ademas ya viene con splits de train, validation y test, cosa que metodologicamente me ordena bastante.

quiero que analices esa eleccion con criterio academico, no solo practico. o sea: necesito saber si ese dataset me sirve para defender bien el trabajo en terminos de modelo, complejidad del problema, posibilidad de experimentar hiperparametros, capacidad de mostrar overfitting y claridad de las metricas finales. no me interesa que me mandes al dataset mas grande o mas de moda; me interesa uno que tenga sentido para esta consigna puntual.

me gustaria que me respondas pensando como si fueras un ayudante medio exigente. decime que ventajas concretas tiene elegir rotten tomatoes para una mlp, pero tambien que limitaciones o flancos criticables podria tener. por ejemplo, si es demasiado simple, si al ser texto corto puede dejar algunas discusiones afuera, o si el hecho de que sea binario y balanceado hace que algunas metricas sean menos interesantes para discutir.

si le ves debilidades, no las tapes: marcame cuales son. pero tambien explicame si esas debilidades son realmente graves para el objetivo del tp o si se pueden defender honestamente diciendo "elegimos este dataset porque nos permitia concentrarnos en entender y comparar la MLP". en otras palabras, necesito una devolucion util para tomar la decision y para poder justificarla despues sin chamuyo.
```

## Prompt 3 - definir una estructura de repo razonable

```text
quiero organizar un proyecto chico de deep learning para un tp, pero no quiero que quede como un rejunte de archivos sueltos. voy a entrenar una mlp para clasificacion binaria con keras sequential y necesito separar de forma prolija las partes importantes: configuracion, carga de datos, construccion del modelo, entrenamiento, evaluacion y resultados.

ayudame a pensar una estructura de carpetas y archivos que sea simple pero seria. no me interesa una arquitectura empresarial gigante, con capas y abstractions innecesarias para un tp, pero tampoco quiero tener todo metido en un solo script porque despues es incomodo de leer, mantener y explicar. la idea es que cualquier compañero, docente o revisor pueda abrir el repo y ubicar rapido donde esta cada cosa.

me gustaria que me propongas nombres concretos de archivos y que me digas que responsabilidad tendria cada uno. por ejemplo, donde conviene definir los experimentos, donde dejar funciones de metricas, donde construir el vectorizador y el modelo, donde poner un smoke test o una prueba minima de entorno, y donde guardar artefactos como curvas de aprendizaje, matriz de confusion o tablas de resultados.

ademas, quiero que me expliques por que esa estructura ayuda a que el trabajo se vea mas ordenado y defendible. si hay algun error tipico de proyectos academicos chicos, como mezclar entrenamiento con evaluacion o dejar hardcodeados demasiados valores en el script principal, me sirve que tambien me lo señales asi lo evito desde el arranque.
```
