from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

from config import DATASET_NAME, RANDOM_SEED


@dataclass(frozen=True)
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _to_numpy(split) -> tuple[np.ndarray, np.ndarray]:
    texts = np.array(split["text"], dtype=object)
    labels = np.array(split["label"], dtype=np.int32)
    return texts, labels


def _limit_balanced(
    texts: np.ndarray,
    labels: np.ndarray,
    limit: int | None,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or limit >= len(labels):
        return texts, labels

    rng = np.random.default_rng(seed)
    per_class = limit // 2
    selected = np.concatenate(
        [
            rng.choice(np.flatnonzero(labels == label), size=per_class, replace=False)
            for label in np.unique(labels)
        ]
    )
    rng.shuffle(selected)
    return texts[selected], labels[selected]


def load_rotten_tomatoes(train_limit: int | None = None) -> DatasetSplits:
    dataset = load_dataset(DATASET_NAME)

    x_train, y_train = _to_numpy(dataset["train"])
    x_val, y_val = _to_numpy(dataset["validation"])
    x_test, y_test = _to_numpy(dataset["test"])

    x_train, y_train = _limit_balanced(x_train, y_train, train_limit)

    return DatasetSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
