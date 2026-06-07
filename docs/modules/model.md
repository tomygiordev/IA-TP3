# `src/model.py`

## Responsabilidad

Construye el vectorizador y el modelo MLP.

## Componentes

- `set_global_determinism`: fija semilla y pide determinismo a TensorFlow.
- `build_vectorizer`: crea `TextVectorization` en modo `tf_idf`.
- `build_mlp`: arma y compila el modelo Keras Sequential.

## Arquitectura base

```text
Input texto
TextVectorization
Dense ReLU
Dropout opcional
Dense sigmoid
```

## Decision defendible

`TextVectorization` se adapta con entrenamiento y queda dentro del modelo. Eso permite guardar o inspeccionar el pipeline completo como una sola arquitectura Keras.
