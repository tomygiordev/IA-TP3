from dataclasses import dataclass


DATASET_NAME = "cornell-movie-review-data/rotten_tomatoes"
RANDOM_SEED = 42
EXPERIMENT_SEEDS = (7, 21, 42, 84, 126, 168, 210, 252, 314, 420)
QUICK_SEEDS = (7, 42, 126)
RESULTS_DIR = "results"
DEMOS_DIR = "results/demos"
THRESHOLD = 0.5

FINALIST_NAMES = (
    "gen2_dropout03_valloss",
    "gen2_dropout01_valloss",
    "gen2_dropout02_valloss",
    "gen_es_valloss",
    "gen2_unigrams_valloss",
    "gen2_unigrams_dropout02",
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    max_tokens: int = 5_000
    hidden_units: tuple[int, ...] = (32,)
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    momentum: float = 0.0
    batch_size: int = 32
    epochs: int = 12
    dropout: float = 0.0
    l2: float = 0.0
    ngrams: int | tuple[int, ...] | None = None
    normalize_tfidf: bool = False
    train_limit: int | None = None
    early_stopping_monitor: str = "val_f1"
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

SGD_EXPERIMENTS = [
    ExperimentConfig(
        name="sgd_no_momentum",
        optimizer="sgd",
        learning_rate=1e-2,
        epochs=20,
        description="SGD sin momentum.",
    ),
    ExperimentConfig(
        name="sgd_momentum_09",
        optimizer="sgd",
        learning_rate=1e-2,
        momentum=0.9,
        epochs=20,
        description="SGD con momentum 0.9.",
    ),
    ExperimentConfig(
        name="sgd_lr_low",
        optimizer="sgd",
        learning_rate=1e-3,
        momentum=0.9,
        epochs=20,
        description="SGD con momentum 0.9 y learning rate menor.",
    ),
    ExperimentConfig(
        name="sgd_lr_high",
        optimizer="sgd",
        learning_rate=1e-1,
        momentum=0.9,
        epochs=20,
        description="SGD con momentum 0.9 y learning rate mayor.",
    ),
    ExperimentConfig(
        name="sgd_dropout",
        optimizer="sgd",
        learning_rate=1e-2,
        momentum=0.9,
        dropout=0.2,
        epochs=20,
        description="SGD con momentum 0.9 y dropout 0.2.",
    ),
    ExperimentConfig(
        name="sgd_l2",
        optimizer="sgd",
        learning_rate=1e-2,
        momentum=0.9,
        l2=1e-4,
        epochs=20,
        description="SGD con momentum 0.9 y regularizacion L2.",
    ),
]

ALL_EXPERIMENTS = [*EXPERIMENTS, *SGD_EXPERIMENTS]


OVERFITTING_EXPERIMENT = ExperimentConfig(
    name="overfitting_demo",
    hidden_units=(512,),
    epochs=35,
    train_limit=500,
    description="Simulacion intencional con pocos datos, mucha capacidad y sin early stopping.",
)

# Base: e9_mlp32_bigrams (max_tokens=10k, ngrams=2, normalize, hidden=(32,), adam lr=1e-3)
# Cada experimento cambia UNA variable para aislar su efecto sobre el overfitting.
GENERALIZATION_EXPERIMENTS = [
    ExperimentConfig(
        name="gen_tokens_1k",
        max_tokens=1_000,
        ngrams=2,
        normalize_tfidf=True,
        description="Vocabulario reducido a 1000 tokens: menos ruido de palabras raras.",
    ),
    ExperimentConfig(
        name="gen_tokens_2k",
        max_tokens=2_000,
        ngrams=2,
        normalize_tfidf=True,
        description="Vocabulario reducido a 2000 tokens: punto medio entre ruido y cobertura.",
    ),
    ExperimentConfig(
        name="gen_tokens_5k",
        max_tokens=5_000,
        ngrams=2,
        normalize_tfidf=True,
        description="Vocabulario de 5000 tokens: compromiso entre el actual (10k) y los reducidos.",
    ),
    ExperimentConfig(
        name="gen_unigrams",
        max_tokens=10_000,
        ngrams=1,
        normalize_tfidf=True,
        description="Unigramas en lugar de bigramas: menor dimensionalidad, menos overfitting.",
    ),
    ExperimentConfig(
        name="gen_units_16",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        hidden_units=(16,),
        description="Capa oculta de 16 neuronas: menos parametros que el ganador (32).",
    ),
    ExperimentConfig(
        name="gen_linear",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        hidden_units=(),
        description="Modelo lineal sin capa oculta: frontera logistica directa sobre TF-IDF.",
    ),
    ExperimentConfig(
        name="gen_l2_001",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        l2=1e-3,
        description="Regularizacion L2 fuerte (0.001): penaliza pesos grandes.",
    ),
    ExperimentConfig(
        name="gen_dropout_02",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        dropout=0.2,
        description="Dropout 0.2: apaga neuronas en entrenamiento para reducir dependencia.",
    ),
    ExperimentConfig(
        name="gen_lr_0005",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        learning_rate=5e-4,
        description="Adam con learning rate 0.0005: aprendizaje mas lento y estable.",
    ),
    ExperimentConfig(
        name="gen_es_valloss",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        early_stopping_monitor="val_loss",
        description="Early stopping sobre val_loss: frena cuando empeora calibracion.",
    ),
]

# Base identica a GENERALIZATION_EXPERIMENTS. Combina las mejores pistas del batch 1.
GENERALIZATION_EXPERIMENTS_2 = [
    ExperimentConfig(
        name="gen2_dropout02_valloss",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        dropout=0.2,
        early_stopping_monitor="val_loss",
        description="Dropout 0.2 + early stopping val_loss.",
    ),
    ExperimentConfig(
        name="gen2_dropout01_valloss",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        dropout=0.1,
        early_stopping_monitor="val_loss",
        description="Dropout 0.1 + early stopping val_loss.",
    ),
    ExperimentConfig(
        name="gen2_dropout03_valloss",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        dropout=0.3,
        early_stopping_monitor="val_loss",
        description="Dropout 0.3 + early stopping val_loss.",
    ),
    ExperimentConfig(
        name="gen2_unigrams_dropout02",
        max_tokens=10_000,
        ngrams=1,
        normalize_tfidf=True,
        dropout=0.2,
        description="Unigramas + dropout 0.2.",
    ),
    ExperimentConfig(
        name="gen2_unigrams_valloss",
        max_tokens=10_000,
        ngrams=1,
        normalize_tfidf=True,
        early_stopping_monitor="val_loss",
        description="Unigramas + early stopping val_loss.",
    ),
    ExperimentConfig(
        name="gen2_l2_1e4",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        l2=1e-4,
        description="L2=0.0001: regularizacion suave.",
    ),
    ExperimentConfig(
        name="gen2_l2_1e5",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        l2=1e-5,
        description="L2=0.00001: regularizacion muy suave.",
    ),
    ExperimentConfig(
        name="gen2_dropout02_l2_1e4",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        dropout=0.2,
        l2=1e-4,
        description="Dropout 0.2 + L2=0.0001.",
    ),
    ExperimentConfig(
        name="gen2_dropout02_l2_1e5",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        dropout=0.2,
        l2=1e-5,
        description="Dropout 0.2 + L2=0.00001.",
    ),
    ExperimentConfig(
        name="gen2_dropout02_valloss_lr0005",
        max_tokens=10_000,
        ngrams=2,
        normalize_tfidf=True,
        dropout=0.2,
        early_stopping_monitor="val_loss",
        learning_rate=5e-4,
        description="Dropout 0.2 + early stopping val_loss + Adam lr=0.0005.",
    ),
]

ALL_GENERALIZATION_EXPERIMENTS = [*GENERALIZATION_EXPERIMENTS, *GENERALIZATION_EXPERIMENTS_2]
