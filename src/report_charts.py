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


def main() -> None:
    results_dir = Path(RESULTS_DIR)
    metrics = pd.read_csv(results_dir / "metrics.csv").sort_values("experiment")

    save_precision_recall_chart(metrics, results_dir / "precision_recall_test.png")
    save_val_test_f1_chart(metrics, results_dir / "val_vs_test_f1.png")

    print(f"Graficos guardados en {results_dir.resolve()}")


if __name__ == "__main__":
    main()
