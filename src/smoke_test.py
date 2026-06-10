from __future__ import annotations

import numpy as np
from tensorflow import keras

from config import ExperimentConfig
from data import load_rotten_tomatoes
from evaluate import classification_metrics, predict_classes
from model import build_mlp, build_normalizer, build_vectorizer, set_global_determinism


def main() -> None:
    set_global_determinism()
    config = ExperimentConfig(
        name="smoke_test",
        max_tokens=1_000,
        hidden_units=(16,),
        epochs=1,
        batch_size=32,
        train_limit=300,
    )
    splits = load_rotten_tomatoes(train_limit=config.train_limit)
    vectorizer = build_vectorizer(config.max_tokens, splits.x_train)
    model = build_mlp(config, preprocessor=vectorizer)

    # Los splits de rotten_tomatoes vienen ordenados por etiqueta: un corte
    # [:n] tomaria una sola clase, el paso [::5] toma ejemplos de ambas.
    x_val, y_val = splits.x_val[::5], splits.y_val[::5]
    x_eval, y_eval = splits.x_test[::5], splits.y_test[::5]

    model.fit(
        splits.x_train,
        splits.y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=2,
    )
    predictions = predict_classes(model, x_eval)
    metrics = classification_metrics(y_eval, predictions)
    print(metrics)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert predictions.shape == y_eval.shape

    numeric_x = np.array(
        [[0.0, 0.1], [0.2, 0.0], [0.8, 1.0], [1.0, 0.9]],
        dtype=np.float32,
    )
    numeric_y = np.array([0, 0, 1, 1], dtype=np.int32)
    normalizer = build_normalizer(numeric_x)
    numeric_model = build_mlp(
        ExperimentConfig(name="numeric_smoke", hidden_units=(4,), epochs=1),
        input_shape=(numeric_x.shape[1],),
        input_dtype="float32",
        preprocessor=normalizer,
    )
    numeric_model.fit(numeric_x, numeric_y, epochs=1, verbose=0)
    numeric_probabilities = numeric_model.predict(numeric_x, verbose=0).reshape(-1)

    assert numeric_probabilities.shape == numeric_y.shape
    assert np.all((0.0 <= numeric_probabilities) & (numeric_probabilities <= 1.0))

    sgd_config = ExperimentConfig(
        name="sgd_smoke",
        hidden_units=(4,),
        optimizer="sgd",
        learning_rate=0.01,
        momentum=0.9,
        epochs=1,
    )
    sgd_model = build_mlp(
        sgd_config,
        input_shape=(numeric_x.shape[1],),
        input_dtype="float32",
        preprocessor=normalizer,
    )
    assert isinstance(sgd_model.optimizer, keras.optimizers.SGD)
    assert np.isclose(float(sgd_model.optimizer.momentum), 0.9)
    sgd_model.fit(numeric_x, numeric_y, epochs=1, verbose=0)
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
