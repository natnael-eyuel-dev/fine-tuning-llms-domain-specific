from __future__ import annotations

"""Central configuration for training/evaluation runs."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    # Dataset
    # NOTE: 'pubmed_rct' is not a valid HF dataset id; use a Hub dataset repo id instead.
    dataset_name: str = "OxAISH-AL-LLM/pubmed_20k_rct"
    dataset_config: str | None = None
    text_field: str | None = None  # auto-detect if None
    label_field: str = "label"

    # Subsampling (useful for CPU / quick runs)
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_test_samples: int | None = None

    # Tokenization
    model_name: str = "distilbert-base-uncased"
    max_length: int = 256

    # Training
    output_dir: Path = Path("artifacts/distilbert_pubmed_rct")
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    logging_steps: int = 50
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    seed: int = 42

    # Early stopping (aligns with "monitor loss, use early stopping")
    early_stopping_patience: int | None = 2
    early_stopping_threshold: float = 0.0

    # Baseline
    baseline_max_features: int = 60_000


