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
    model.fit(
        splits.x_train,
        splits.y_train,
        validation_data=(splits.x_val[:200], splits.y_val[:200]),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=2,
    )
    predictions = predict_classes(model, splits.x_test[:200])
    metrics = classification_metrics(splits.y_test[:200], predictions)
    print(metrics)


if __name__ == "__main__":
    main()
