from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from config import ExperimentConfig, RANDOM_SEED


def set_global_determinism(seed: int = RANDOM_SEED) -> None:
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_vectorizer(max_tokens: int, train_texts) -> layers.TextVectorization:
    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="tf_idf",
        standardize="lower_and_strip_punctuation",
        name="text_vectorization",
    )
    vectorizer.adapt(train_texts)
    return vectorizer


def build_mlp(config: ExperimentConfig, vectorizer: layers.TextVectorization) -> keras.Model:
    l2_regularizer = regularizers.l2(config.l2) if config.l2 else None

    model_layers: list[keras.layers.Layer] = [
        keras.Input(shape=(), dtype=tf.string, name="text"),
        vectorizer,
    ]

    for index, units in enumerate(config.hidden_units, start=1):
        model_layers.append(
            layers.Dense(
                units,
                activation="relu",
                kernel_regularizer=l2_regularizer,
                name=f"dense_hidden_{index}",
            )
        )
        if config.dropout:
            model_layers.append(layers.Dropout(config.dropout, name=f"dropout_{index}"))

    model_layers.append(layers.Dense(1, activation="sigmoid", name="sentiment_probability"))

    model = keras.Sequential(model_layers, name=config.name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
