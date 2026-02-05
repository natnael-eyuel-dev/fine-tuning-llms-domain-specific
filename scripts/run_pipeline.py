from __future__ import annotations

"""One-command pipeline: dataset -> baseline -> fine-tune -> metrics artifact."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baseline import train_tfidf_logreg_baseline
from src.config import TrainConfig
from src.data import load_hf_classification_dataset
from src.finetune import finetune_distilbert_classifier
from src.reporting import write_json


def main() -> None:
    cfg = TrainConfig()

    # Make CPU runs feasible by default
    if not torch.cuda.is_available():
        cfg = replace(cfg, max_train_samples=2000, max_eval_samples=1000, max_test_samples=1000)

    ds, text_field, id2label = load_hf_classification_dataset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        text_field=cfg.text_field,
        label_field=cfg.label_field,
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples,
        max_test_samples=cfg.max_test_samples,
        seed=cfg.seed,
    )

    baseline_result, baseline_metrics = train_tfidf_logreg_baseline(
        ds,
        text_field=text_field,
        label_field=cfg.label_field,
        max_features=cfg.baseline_max_features,
        seed=cfg.seed,
    )

    ft_result, ft_metrics = finetune_distilbert_classifier(
        ds,
        text_field=text_field,
        label_field=cfg.label_field,
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        seed=cfg.seed,
        id2label=id2label,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        early_stopping_patience=cfg.early_stopping_patience,
        early_stopping_threshold=cfg.early_stopping_threshold,
    )

    metrics_path = ROOT / "artifacts" / "metrics.json"
    write_json(
        metrics_path,
        {
            "dataset": {
                "name": cfg.dataset_name,
                "config": cfg.dataset_config,
                "text_field": text_field,
                "splits": {k: len(v) for k, v in ds.items()},
            },
            "baseline": {"test_accuracy": baseline_result.accuracy, "test_macro_f1": baseline_result.macro_f1, **baseline_metrics},
            "finetune": {
                "model_dir": str(ft_result.model_dir),
                "test_accuracy": ft_result.test_accuracy,
                "test_macro_f1": ft_result.test_macro_f1,
                **ft_metrics,
            },
        },
    )

    print("Wrote:", metrics_path)
    print(json.dumps(json.loads(metrics_path.read_text(encoding="utf-8")), indent=2)[:2000])


if __name__ == "__main__":
    main()


