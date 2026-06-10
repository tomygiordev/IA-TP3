from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from config import ExperimentConfig, RANDOM_SEED


def set_global_determinism(seed: int = RANDOM_SEED) -> None:
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_vectorizer(
    max_tokens: int,
    train_texts,
    ngrams: int | tuple[int, ...] | None = None,
) -> layers.TextVectorization:
    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="tf_idf",
        ngrams=ngrams,
        standardize="lower_and_strip_punctuation",
        name="text_vectorization",
    )
    vectorizer.adapt(train_texts)
    return vectorizer


def build_normalizer(train_features) -> layers.Normalization:
    normalizer = layers.Normalization(name="numeric_normalization")
    normalizer.adapt(train_features)
    return normalizer


class BinaryF1(keras.metrics.Metric):
    def __init__(self, name: str = "f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.bool)
        y_pred = tf.reshape(y_pred, (-1,)) > 0.5
        weights = 1.0 if sample_weight is None else tf.cast(sample_weight, self.dtype)

        self.true_positives.assign_add(
            tf.reduce_sum(tf.cast(y_true & y_pred, self.dtype) * weights)
        )
        self.false_positives.assign_add(
            tf.reduce_sum(tf.cast(~y_true & y_pred, self.dtype) * weights)
        )
        self.false_negatives.assign_add(
            tf.reduce_sum(tf.cast(y_true & ~y_pred, self.dtype) * weights)
        )

    def result(self):
        return tf.math.divide_no_nan(
            2.0 * self.true_positives,
            2.0 * self.true_positives + self.false_positives + self.false_negatives,
        )

    def reset_state(self):
        for variable in self.variables:
            variable.assign(0.0)


def build_mlp(
    config: ExperimentConfig,
    *,
    input_shape: tuple[int, ...] = (),
    input_dtype: str = "string",
    preprocessor: keras.layers.Layer | None = None,
) -> keras.Model:
    l2_regularizer = regularizers.l2(config.l2) if config.l2 else None

    model_layers: list[keras.layers.Layer] = [
        keras.Input(shape=input_shape, dtype=input_dtype, name="input")
    ]
    if preprocessor is not None:
        model_layers.append(preprocessor)

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

    model_layers.append(layers.Dense(1, activation="sigmoid", name="class_probability"))

    model = keras.Sequential(model_layers, name=config.name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            BinaryF1(),
        ],
    )
    return model
