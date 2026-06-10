from __future__ import annotations

from config import ExperimentConfig
from data import load_rotten_tomatoes
from evaluate import classification_metrics, predict_classes
from model import build_mlp, build_vectorizer, set_global_determinism


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
    model = build_mlp(config, vectorizer)

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
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
