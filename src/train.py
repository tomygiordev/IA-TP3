from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from config import (
    EXPERIMENT_SEEDS,
    EXPERIMENTS,
    OVERFITTING_EXPERIMENT,
    RANDOM_SEED,
    RESULTS_DIR,
    THRESHOLD,
)
from data import load_rotten_tomatoes
from evaluate import (
    classification_metrics,
    find_best_f1_threshold,
    predict_probabilities,
    save_confusion_matrix,
    save_learning_curves,
)
from model import build_mlp, build_vectorizer, set_global_determinism


def build_text_preprocessor(config, train_texts):
    vectorizer = build_vectorizer(config.max_tokens, train_texts, config.ngrams)
    if not config.normalize_tfidf:
        return vectorizer
    return keras.Sequential(
        [vectorizer, keras.layers.UnitNormalization(axis=-1)],
        name="text_preprocessing",
    )


def train_experiment(config, seed: int, *, early_stopping: bool = True, verbose: int = 0):
    keras.backend.clear_session()
    set_global_determinism(seed)
    splits = load_rotten_tomatoes(train_limit=config.train_limit)
    preprocessor = build_text_preprocessor(config, splits.x_train)
    model = build_mlp(config, preprocessor=preprocessor)

    callbacks = []
    if early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_f1",
                mode="max",
                patience=5,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        splits.x_train,
        splits.y_train,
        validation_data=(splits.x_val, splits.y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    val_probabilities = predict_probabilities(model, splits.x_val)
    val_predictions = (val_probabilities > THRESHOLD).astype(np.int32)
    val_metrics = classification_metrics(splits.y_val, val_predictions)
    best_threshold, tuned_val_f1 = find_best_f1_threshold(
        splits.y_val,
        val_probabilities,
    )

    row = {
        "experiment": config.name,
        "description": config.description,
        "seed": seed,
        "max_tokens": config.max_tokens,
        "ngrams": config.ngrams or 1,
        "normalize_tfidf": config.normalize_tfidf,
        "hidden_units": str(config.hidden_units),
        "parameters": model.count_params(),
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "epochs_requested": config.epochs,
        "epochs_ran": len(history.history["loss"]),
        "dropout": config.dropout,
        "l2": config.l2,
        "train_limit": config.train_limit or "all",
        "best_threshold": best_threshold,
        "val_f1_tuned": tuned_val_f1,
        **{f"val_{key}": value for key, value in val_metrics.items()},
    }
    return row, model, history, splits


def bootstrap_mean_ci(values: pd.Series, seed: int = RANDOM_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values.to_numpy(), size=(10_000, len(values)), replace=True)
    return tuple(np.quantile(samples.mean(axis=1), [0.025, 0.975]))


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, group in runs.groupby("experiment", sort=False):
        ci_low, ci_high = bootstrap_mean_ci(group["val_f1_tuned"])
        first = group.iloc[0]
        rows.append(
            {
                "experiment": experiment,
                "description": first["description"],
                "hidden_units": first["hidden_units"],
                "parameters": int(first["parameters"]),
                "dropout": first["dropout"],
                "l2": first["l2"],
                "ngrams": first["ngrams"],
                "normalize_tfidf": first["normalize_tfidf"],
                "val_f1_tuned_mean": group["val_f1_tuned"].mean(),
                "val_f1_tuned_std": group["val_f1_tuned"].std(ddof=1),
                "val_f1_ci_low": ci_low,
                "val_f1_ci_high": ci_high,
                "val_f1_at_05_mean": group["val_f1"].mean(),
                "threshold_mean": group["best_threshold"].mean(),
                "epochs_mean": group["epochs_ran"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["val_f1_tuned_mean", "val_f1_tuned_std", "parameters"],
        ascending=[False, True, True],
    )


def evaluate_final_model(config):
    row, model, history, splits = train_experiment(
        config,
        RANDOM_SEED,
        verbose=2,
    )
    test_probabilities = predict_probabilities(model, splits.x_test)
    test_predictions = (test_probabilities > row["best_threshold"]).astype(np.int32)
    test_metrics = classification_metrics(splits.y_test, test_predictions)
    final_row = {
        **row,
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    return final_row, history, splits, test_predictions


def main() -> None:
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    runs_path = results_dir / "runs.csv"

    if runs_path.exists():
        runs = pd.read_csv(runs_path)
        run_rows = runs.to_dict("records")
        completed = set(zip(runs["experiment"], runs["seed"]))
        print(f"Reanudando desde {len(run_rows)} corridas guardadas")
    else:
        run_rows = []
        completed = set()

    for config in EXPERIMENTS:
        for seed in EXPERIMENT_SEEDS:
            if (config.name, seed) in completed:
                continue
            print(f"Entrenando {config.name} con semilla {seed}")
            row, _, _, _ = train_experiment(config, seed)
            run_rows.append(row)
            pd.DataFrame(run_rows).to_csv(runs_path, index=False)

    runs = pd.DataFrame(run_rows)
    summary = summarize_runs(runs)
    summary.to_csv(results_dir / "summary.csv", index=False)

    best_name = summary.iloc[0]["experiment"]
    best_config = next(config for config in EXPERIMENTS if config.name == best_name)
    final_row, final_history, splits, test_predictions = evaluate_final_model(best_config)
    pd.DataFrame([final_row]).to_csv(results_dir / "final_metrics.csv", index=False)
    save_confusion_matrix(
        splits.y_test,
        test_predictions,
        results_dir / "confusion_matrix.png",
    )
    save_learning_curves(final_history, results_dir / "best_learning_curves.png")

    _, _, overfitting_history, _ = train_experiment(
        OVERFITTING_EXPERIMENT,
        RANDOM_SEED,
        early_stopping=False,
        verbose=2,
    )
    save_learning_curves(
        overfitting_history,
        results_dir / "overfitting_learning_curves.png",
    )

    print("\nMejor configuracion por F1 medio de validacion ajustado:")
    print(summary.iloc[0].to_string())
    print("\nEvaluacion final en test:")
    print(pd.Series(final_row).to_string())
    print(f"\nResultados guardados en {results_dir.resolve()}")


if __name__ == "__main__":
    main()
