from dataclasses import dataclass


DATASET_NAME = "cornell-movie-review-data/rotten_tomatoes"
RANDOM_SEED = 42
RESULTS_DIR = "results"
THRESHOLD = 0.5


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    max_tokens: int = 5_000
    hidden_units: tuple[int, ...] = (32,)
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 12
    dropout: float = 0.0
    l2: float = 0.0
    train_limit: int | None = None
    description: str = ""


EXPERIMENTS = [
    ExperimentConfig(
        name="e0_baseline",
        hidden_units=(32,),
        description="Modelo simple inicial.",
    ),
    ExperimentConfig(
        name="e1_more_capacity",
        hidden_units=(128, 64),
        description="Mayor capacidad: mas neuronas y dos capas ocultas.",
    ),
    ExperimentConfig(
        name="e2_lower_learning_rate",
        hidden_units=(64,),
        learning_rate=1e-4,
        description="Menor tasa de aprendizaje.",
    ),
    ExperimentConfig(
        name="e3_regularized",
        hidden_units=(128, 64),
        dropout=0.5,
        l2=1e-4,
        description="Regularizacion con dropout y L2.",
    ),
    ExperimentConfig(
        name="e4_overfitting_demo",
        hidden_units=(512, 512, 256),
        epochs=35,
        train_limit=500,
        description="Simulacion intencional de overfitting con pocos datos y mucha capacidad.",
    ),
]
