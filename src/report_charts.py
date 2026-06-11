from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR


def save_precision_recall_chart(metrics: pd.DataFrame, output_path: str | Path) -> None:
    positions = np.arange(len(metrics))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(positions - width / 2, metrics["test_precision"], width, label="precision")
    ax.bar(positions + width / 2, metrics["test_recall"], width, label="recall")
    ax.set_xticks(positions, metrics["experiment"], rotation=15, ha="right")
    ax.set_ylim(0.5, 0.85)
    ax.set_title("Precision vs recall en test (umbral 0.5)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_val_test_f1_chart(metrics: pd.DataFrame, output_path: str | Path) -> None:
    positions = np.arange(len(metrics))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(positions - width / 2, metrics["val_f1"], width, label="val_f1")
    ax.bar(positions + width / 2, metrics["test_f1"], width, label="test_f1")
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="nivel de azar")
    ax.set_xticks(positions, metrics["experiment"], rotation=15, ha="right")
    ax.set_ylim(0.4, 0.85)
    ax.set_title("F1 de validacion vs test por experimento")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_finalists_learning_curves(curves_df: pd.DataFrame, output_path: str | Path) -> None:
    experiments = curves_df["experiment"].unique()
    n = len(experiments)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for i, exp in enumerate(experiments):
        ax = axes[i]
        exp_data = curves_df[curves_df["experiment"] == exp]
        avg = exp_data.groupby("epoch")[["train_loss", "val_loss"]].mean()
        std_val = exp_data.groupby("epoch")["val_loss"].std().fillna(0)

        ax.plot(avg.index, avg["train_loss"], label="train", color="steelblue")
        ax.plot(avg.index, avg["val_loss"], label="val", color="orange")
        ax.fill_between(avg.index, avg["val_loss"] - std_val, avg["val_loss"] + std_val, alpha=0.2, color="orange")
        ax.set_title(exp, fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Learning curves promediadas (10 semillas) — finalistas", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_finalists_comparison_chart(summary: pd.DataFrame, output_path: str | Path) -> None:
    summary = summary.sort_values("composite_score", ascending=True)
    experiments = summary["experiment"].tolist()
    y = np.arange(len(experiments))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].barh(y, summary["val_f1_tuned_mean"], xerr=summary["val_f1_tuned_std"],
                 color="#4caf73", capsize=4, height=0.6)
    axes[0].set_yticks(y, experiments, fontsize=8)
    axes[0].set_xlabel("val_f1_tuned_mean")
    axes[0].set_title("F1 validacion (ajustado)")

    axes[1].barh(y, summary["val_loss_mean"], xerr=summary["val_loss_std"],
                 color="#5b8dd9", capsize=4, height=0.6)
    axes[1].set_yticks(y, [""] * len(experiments))
    axes[1].set_xlabel("val_loss_mean")
    axes[1].set_title("Val loss")

    colors_score = plt.cm.RdYlGn(
        (summary["composite_score"] - summary["composite_score"].min()) /
        (summary["composite_score"].max() - summary["composite_score"].min() + 1e-12)
    )
    axes[2].barh(y, summary["composite_score"], color=colors_score, height=0.6)
    axes[2].set_yticks(y, [""] * len(experiments))
    axes[2].set_xlabel("Score compuesto")
    axes[2].set_title("Score compuesto (mayor = mejor)")

    plt.suptitle("Comparacion de finalistas — solo validacion", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_generalization_chart(summary: pd.DataFrame, baseline_f1: float, output_path: str | Path) -> None:
    summary = summary.sort_values("val_f1_tuned_mean", ascending=True)
    experiments = summary["experiment"].tolist()
    means = summary["val_f1_tuned_mean"].tolist()
    stds = summary["val_f1_tuned_std"].tolist()
    val_losses = summary["val_loss_best_mean"].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#e05c5c" if m < baseline_f1 else "#4caf73" for m in means]
    bars = ax1.barh(experiments, means, xerr=stds, color=colors, capsize=4, height=0.6)
    ax1.axvline(baseline_f1, linestyle="--", color="#333", linewidth=1.2, label=f"ganador actual ({baseline_f1:.4f})")
    ax1.set_xlabel("val_f1_tuned_mean (3 semillas)")
    ax1.set_title("F1 validacion: experimentos de generalizacion")
    ax1.legend(fontsize=8)
    ax1.set_xlim(min(means) - 0.03, max(means) + 0.03)

    colors2 = ["#e05c5c" if vl > 0.52 else "#4caf73" for vl in val_losses]
    ax2.barh(experiments, val_losses, color=colors2, height=0.6)
    ax2.set_xlabel("val_loss_best_mean (3 semillas)")
    ax2.set_title("Val loss: experimentos de generalizacion")
    ax2.set_xlim(min(val_losses) - 0.03, max(val_losses) + 0.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    results_dir = Path(RESULTS_DIR)
    metrics = pd.read_csv(results_dir / "metrics.csv").sort_values("experiment")

    save_precision_recall_chart(metrics, results_dir / "precision_recall_test.png")
    save_val_test_f1_chart(metrics, results_dir / "val_vs_test_f1.png")

    gen_summary_path = results_dir / "generalization_summary.csv"
    if gen_summary_path.exists():
        gen_summary = pd.read_csv(gen_summary_path)
        original_summary = pd.read_csv(results_dir / "summary.csv")
        baseline_f1 = float(original_summary.iloc[0]["val_f1_tuned_mean"])
        save_generalization_chart(gen_summary, baseline_f1, results_dir / "generalization_comparison.png")
        print("Grafico de generalizacion guardado.")

    finalists_summary_path = results_dir / "finalists_summary.csv"
    finalists_curves_path = results_dir / "finalists_curves.csv"
    if finalists_summary_path.exists():
        fin_summary = pd.read_csv(finalists_summary_path)
        save_finalists_comparison_chart(fin_summary, results_dir / "finalists_comparison.png")
        print("Grafico de comparacion de finalistas guardado.")
    if finalists_curves_path.exists():
        curves_df = pd.read_csv(finalists_curves_path)
        save_finalists_learning_curves(curves_df, results_dir / "finalists_learning_curves.png")
        print("Learning curves de finalistas guardadas.")

    print(f"Graficos guardados en {results_dir.resolve()}")


if __name__ == "__main__":
    main()
