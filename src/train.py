from __future__ import annotations

from pathlib import Path

import pandas as pd
from tensorflow import keras

from config import EXPERIMENTS, RESULTS_DIR
from data import load_rotten_tomatoes
from evaluate import (
    classification_metrics,
    predict_classes,
    save_confusion_matrix,
    save_learning_curves,
)
from model import build_mlp, build_vectorizer, set_global_determinism


def train_experiment(config):
    splits = load_rotten_tomatoes(train_limit=config.train_limit)
    vectorizer = build_vectorizer(config.max_tokens, splits.x_train)
    model = build_mlp(config, vectorizer)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    if "overfitting" in config.name:
        callbacks = []

    history = model.fit(
        splits.x_train,
        splits.y_train,
        validation_data=(splits.x_val, splits.y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    val_pred, _ = predict_classes(model, splits.x_val)
    test_pred, _ = predict_classes(model, splits.x_test)

    val_metrics = classification_metrics(splits.y_val, val_pred)
    test_metrics = classification_metrics(splits.y_test, test_pred)

    row = {
        "experiment": config.name,
        "description": config.description,
        "max_tokens": config.max_tokens,
        "hidden_units": str(config.hidden_units),
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs_requested": config.epochs,
        "epochs_ran": len(history.history["loss"]),
        "dropout": config.dropout,
        "l2": config.l2,
        "train_limit": config.train_limit or "all",
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }

    return row, model, history, splits, test_pred


def main() -> None:
    set_global_determinism()
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)

    rows = []
    best = None
    overfitting_result = None
    for config in EXPERIMENTS:
        print(f"\n=== Entrenando {config.name} ===")
        row, model, history, splits, test_pred = train_experiment(config)
        rows.append(row)
        if best is None or row["val_f1"] > best["row"]["val_f1"]:
            best = {
                "row": row,
                "model": model,
                "history": history,
                "splits": splits,
                "test_pred": test_pred,
            }
        if "overfitting" in config.name:
            overfitting_result = {"history": history}

    metrics = pd.DataFrame(rows).sort_values("val_f1", ascending=False)
    metrics.to_csv(results_dir / "metrics.csv", index=False)

    save_confusion_matrix(
        best["splits"].y_test,
        best["test_pred"],
        results_dir / "confusion_matrix.png",
    )
    save_learning_curves(best["history"], results_dir / "learning_curves.png")
    save_learning_curves(best["history"], results_dir / "best_learning_curves.png")
    if overfitting_result is not None:
        save_learning_curves(
            overfitting_result["history"],
            results_dir / "overfitting_learning_curves.png",
        )

    print("\nMejor experimento por val_f1:")
    print(metrics.iloc[0].to_string())
    print(f"\nResultados guardados en {results_dir.resolve()}")


if __name__ == "__main__":
    main()
