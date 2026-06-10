from dataclasses import dataclass


DATASET_NAME = "cornell-movie-review-data/rotten_tomatoes"
RANDOM_SEED = 42
EXPERIMENT_SEEDS = (7, 21, 42, 84, 126, 168, 210, 252, 314, 420)
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
    ngrams: int | tuple[int, ...] | None = None
    normalize_tfidf: bool = False
    train_limit: int | None = None
    description: str = ""


EXPERIMENTS = [
    ExperimentConfig(
        name="e0_baseline",
        hidden_units=(32,),
        description="Modelo simple inicial.",
    ),
    ExperimentConfig(
        name="e1_linear",
        hidden_units=(),
        description="Baseline lineal: TF-IDF conectado directamente a la salida.",
    ),
    ExperimentConfig(
        name="e2_more_capacity",
        hidden_units=(128,),
        description="Mayor capacidad: mas neuronas en una sola capa oculta.",
    ),
    ExperimentConfig(
        name="e3_dropout_32",
        dropout=0.2,
        description="Red chica con dropout moderado.",
    ),
    ExperimentConfig(
        name="e4_l2_32",
        l2=1e-4,
        description="Red chica con regularizacion L2.",
    ),
    ExperimentConfig(
        name="e5_dropout_128",
        hidden_units=(128,),
        dropout=0.2,
        description="Red amplia con dropout moderado.",
    ),
    ExperimentConfig(
        name="e6_l2_128",
        hidden_units=(128,),
        l2=1e-4,
        description="Red amplia con regularizacion L2.",
    ),
    ExperimentConfig(
        name="e7_combined_128",
        hidden_units=(128,),
        dropout=0.5,
        l2=1e-4,
        description="Configuracion amplia con dropout y L2 del estudio anterior.",
    ),
    ExperimentConfig(
        name="e8_linear_bigrams",
        hidden_units=(),
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        description="Baseline lineal con unigramas, bigramas y normalizacion L2.",
    ),
    ExperimentConfig(
        name="e9_mlp32_bigrams",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        description="MLP chica con unigramas, bigramas y normalizacion L2.",
    ),
]


OVERFITTING_EXPERIMENT = ExperimentConfig(
    name="overfitting_demo",
    hidden_units=(512,),
    epochs=35,
    train_limit=500,
    description="Simulacion intencional con pocos datos, mucha capacidad y sin early stopping.",
)
