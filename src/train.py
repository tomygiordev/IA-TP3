from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from config import (
    ALL_EXPERIMENTS,
    ALL_GENERALIZATION_EXPERIMENTS,
    DEMOS_DIR,
    EXPERIMENT_SEEDS,
    FINALIST_NAMES,
    OVERFITTING_EXPERIMENT,
    QUICK_SEEDS,
    RANDOM_SEED,
    RESULTS_DIR,
    SGD_EXPERIMENTS,
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

    monitor = config.early_stopping_monitor
    monitor_mode = "min" if monitor == "val_loss" else "max"
    callbacks = []
    if early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode=monitor_mode,
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
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "momentum": config.momentum,
        "batch_size": config.batch_size,
        "epochs_requested": config.epochs,
        "epochs_ran": len(history.history["loss"]),
        "dropout": config.dropout,
        "l2": config.l2,
        "train_limit": config.train_limit or "all",
        "early_stopping_monitor": config.early_stopping_monitor,
        "val_loss_best": float(min(history.history["val_loss"])),
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
                "optimizer": first["optimizer"],
                "learning_rate": first["learning_rate"],
                "momentum": first["momentum"],
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


def build_finalists_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, group in runs.groupby("experiment", sort=False):
        ci_low, ci_high = bootstrap_mean_ci(group["val_f1_tuned"])
        first = group.iloc[0]
        rows.append({
            "experiment": experiment,
            "n_seeds": len(group),
            "parameters": int(first["parameters"]),
            "dropout": first["dropout"],
            "l2": first["l2"],
            "early_stopping_monitor": first["early_stopping_monitor"],
            "ngrams": first["ngrams"],
            "val_f1_tuned_mean": group["val_f1_tuned"].mean(),
            "val_f1_tuned_std": group["val_f1_tuned"].std(ddof=1),
            "val_f1_ci_low": ci_low,
            "val_f1_ci_high": ci_high,
            "val_loss_mean": group["val_loss_best"].mean(),
            "val_loss_std": group["val_loss_best"].std(ddof=1),
            "val_accuracy_mean": group["val_accuracy"].mean(),
            "best_threshold_mean": group["best_threshold"].mean(),
            "epochs_mean": group["epochs_ran"].mean(),
        })
    df = pd.DataFrame(rows)

    def _norm(s: pd.Series, invert: bool = False) -> pd.Series:
        span = s.max() - s.min()
        normalized = (s - s.min()) / (span if span > 1e-12 else 1.0)
        return 1 - normalized if invert else normalized

    df["composite_score"] = (
        _norm(df["val_f1_tuned_mean"]) * 0.45
        + _norm(df["val_f1_tuned_std"], invert=True) * 0.25
        + _norm(df["val_loss_mean"], invert=True) * 0.20
        + _norm(df["val_loss_std"], invert=True) * 0.10
    )
    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


def run_finalists(results_dir: Path) -> None:
    finalist_configs = [c for c in ALL_GENERALIZATION_EXPERIMENTS if c.name in FINALIST_NAMES]

    runs_path = results_dir / "finalists_runs.csv"
    curves_path = results_dir / "finalists_curves.csv"

    if runs_path.exists():
        runs_df = pd.read_csv(runs_path)
        run_rows = runs_df.to_dict("records")
        completed = set(zip(runs_df["experiment"], runs_df["seed"]))
        print(f"Reanudando finalistas desde {len(run_rows)} corridas guardadas")
    else:
        run_rows = []
        completed = set()

    curve_rows = pd.read_csv(curves_path).to_dict("records") if curves_path.exists() else []
    completed_curves = {(r["experiment"], r["seed"]) for r in curve_rows}

    for config in finalist_configs:
        for seed in EXPERIMENT_SEEDS:
            if (config.name, seed) in completed:
                continue
            print(f"[final] {config.name} seed={seed}")
            row, _, history, _ = train_experiment(config, seed)
            run_rows.append(row)

            if (config.name, seed) not in completed_curves:
                for epoch_idx, (tl, vl, ta, va) in enumerate(zip(
                    history.history["loss"],
                    history.history["val_loss"],
                    history.history["accuracy"],
                    history.history["val_accuracy"],
                ), start=1):
                    curve_rows.append({
                        "experiment": config.name,
                        "seed": seed,
                        "epoch": epoch_idx,
                        "train_loss": tl,
                        "val_loss": vl,
                        "train_accuracy": ta,
                        "val_accuracy": va,
                    })

            pd.DataFrame(run_rows).to_csv(runs_path, index=False)
            pd.DataFrame(curve_rows).to_csv(curves_path, index=False)

    runs = pd.DataFrame(run_rows)
    summary = build_finalists_summary(runs)
    summary.to_csv(results_dir / "finalists_summary.csv", index=False)

    print("\n=== Finalistas: tabla comparativa (orden por score compuesto) ===")
    display_cols = [
        "experiment", "n_seeds", "val_f1_tuned_mean", "val_f1_tuned_std",
        "val_loss_mean", "val_loss_std", "val_accuracy_mean",
        "best_threshold_mean", "epochs_mean", "composite_score",
    ]
    print(summary[display_cols].to_string(index=False))

    winner = summary.iloc[0]
    winner_name = winner["experiment"]
    winner_config = next(c for c in finalist_configs if c.name == winner_name)

    _print_selection_rationale(summary, winner_name)

    print(f"\n>>> Evaluando modelo final en TEST: {winner_name}")
    final_row, final_history, splits, test_predictions = evaluate_final_model(winner_config)

    final_row["selection_rationale"] = (
        f"Elegido por score compuesto {winner['composite_score']:.4f}: "
        f"val_f1={winner['val_f1_tuned_mean']:.4f}±{winner['val_f1_tuned_std']:.4f}, "
        f"val_loss={winner['val_loss_mean']:.4f}±{winner['val_loss_std']:.4f}, "
        f"epochs_mean={winner['epochs_mean']:.1f}"
    )
    pd.DataFrame([final_row]).to_csv(results_dir / "final_metrics.csv", index=False)
    save_confusion_matrix(splits.y_test, test_predictions, results_dir / "confusion_matrix.png")
    save_learning_curves(final_history, results_dir / "best_learning_curves.png")

    print("\n=== Metricas finales en TEST ===")
    test_keys = [k for k in final_row if k.startswith("test_")]
    for k in test_keys:
        print(f"  {k}: {final_row[k]:.4f}")
    print(f"  threshold usado: {final_row['best_threshold']:.4f}")
    print(f"\nArchivos guardados en {results_dir.resolve()}")


def _print_selection_rationale(summary: pd.DataFrame, winner_name: str) -> None:
    print(f"\n=== Criterio de seleccion ===")
    print(f"Ganador: {winner_name}")
    print("Score compuesto = 0.45·F1_mean + 0.25·(1-F1_std) + 0.20·(1-loss_mean) + 0.10·(1-loss_std)")
    print("\nComparacion top candidatos:")
    top = summary.head(3)[["experiment", "val_f1_tuned_mean", "val_f1_tuned_std", "val_loss_mean", "composite_score"]]
    print(top.to_string(index=False))

    runner_up = summary.iloc[1]
    delta_f1 = summary.iloc[0]["val_f1_tuned_mean"] - runner_up["val_f1_tuned_mean"]
    pooled_std = (summary.iloc[0]["val_f1_tuned_std"] + runner_up["val_f1_tuned_std"]) / 2
    noise_ratio = abs(delta_f1) / (pooled_std + 1e-12)
    conclusion = "REAL (>1 std)" if noise_ratio > 1.0 else "RUIDO ESTADISTICO (<1 std)"
    print(f"\nDiferencia entre puesto 1 y 2: delta_F1={delta_f1:.4f}, pooled_std={pooled_std:.4f} → {conclusion}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sgd-only",
        action="store_true",
        help="Ejecuta solamente las configuraciones SGD pendientes y omite la demo de overfitting.",
    )
    parser.add_argument(
        "--generalization",
        action="store_true",
        help="Ejecuta los experimentos de generalizacion con 3 semillas y guarda en results/generalization_*.csv.",
    )
    parser.add_argument(
        "--finalists",
        action="store_true",
        help="Protocolo final: 6 candidatos x 10 semillas, scoring compuesto, evaluacion unica en test.",
    )
    return parser.parse_args()


def run_generalization(results_dir: Path) -> None:
    runs_path = results_dir / "generalization_runs.csv"

    if runs_path.exists():
        runs = pd.read_csv(runs_path)
        run_rows = runs.to_dict("records")
        completed = set(zip(runs["experiment"], runs["seed"]))
        print(f"Reanudando generalizacion desde {len(run_rows)} corridas guardadas")
    else:
        run_rows = []
        completed = set()

    for config in ALL_GENERALIZATION_EXPERIMENTS:
        for seed in QUICK_SEEDS:
            if (config.name, seed) in completed:
                continue
            print(f"[gen] Entrenando {config.name} con semilla {seed}")
            row, _, _, _ = train_experiment(config, seed)
            run_rows.append(row)
            pd.DataFrame(run_rows).to_csv(runs_path, index=False)

    runs = pd.DataFrame(run_rows)
    summary = summarize_runs(runs)
    summary["val_loss_best_mean"] = runs.groupby("experiment")["val_loss_best"].mean().reindex(summary["experiment"]).values
    summary.to_csv(results_dir / "generalization_summary.csv", index=False)

    print("\n=== Generalizacion: ranking por val_f1_tuned_mean ===")
    cols = ["experiment", "val_f1_tuned_mean", "val_f1_tuned_std", "val_loss_best_mean", "description"]
    print(summary[cols].to_string(index=False))
    print(f"\nResultados guardados en {runs_path.parent.resolve()}")


def main() -> None:
    args = parse_args()
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)

    if args.generalization:
        run_generalization(results_dir)
        return

    if args.finalists:
        run_finalists(results_dir)
        return

    runs_path = results_dir / "runs.csv"

    if runs_path.exists():
        runs = pd.read_csv(runs_path)
        if "optimizer" not in runs:
            runs["optimizer"] = "adam"
        if "momentum" not in runs:
            runs["momentum"] = 0.0
        run_rows = runs.to_dict("records")
        completed = set(zip(runs["experiment"], runs["seed"]))
        print(f"Reanudando desde {len(run_rows)} corridas guardadas")
    else:
        run_rows = []
        completed = set()

    selected_experiments = SGD_EXPERIMENTS if args.sgd_only else ALL_EXPERIMENTS
    for config in selected_experiments:
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
    best_config = next(config for config in ALL_EXPERIMENTS if config.name == best_name)
    final_row, final_history, splits, test_predictions = evaluate_final_model(best_config)
    pd.DataFrame([final_row]).to_csv(results_dir / "final_metrics.csv", index=False)
    save_confusion_matrix(
        splits.y_test,
        test_predictions,
        results_dir / "confusion_matrix.png",
    )
    save_learning_curves(final_history, results_dir / "best_learning_curves.png")

    if not args.sgd_only:
        demos_dir = Path(DEMOS_DIR)
        demos_dir.mkdir(parents=True, exist_ok=True)
        _, _, overfitting_history, _ = train_experiment(
            OVERFITTING_EXPERIMENT,
            RANDOM_SEED,
            early_stopping=False,
            verbose=2,
        )
        save_learning_curves(
            overfitting_history,
            demos_dir / "overfitting_learning_curves.png",
        )

    print("\nMejor configuracion por F1 medio de validacion ajustado:")
    print(summary.iloc[0].to_string())
    print("\nEvaluacion final en test:")
    print(pd.Series(final_row).to_string())
    print(f"\nResultados guardados en {results_dir.resolve()}")


if __name__ == "__main__":
    main()
