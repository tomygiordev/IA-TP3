from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import THRESHOLD


def predict_classes(model, texts: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    probabilities = model.predict(texts, verbose=0).reshape(-1)
    return (probabilities > threshold).astype(np.int32)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["negativo", "positivo"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de confusion - test")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_learning_curves(history, output_path: str | Path) -> None:
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history_dict["loss"], label="train")
    axes[0].plot(epochs, history_dict["val_loss"], label="validacion")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history_dict["accuracy"], label="train")
    axes[1].plot(epochs, history_dict["val_accuracy"], label="validacion")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
