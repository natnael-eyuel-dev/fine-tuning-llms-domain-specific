from __future__ import annotations

"""Fine-tuning utilities using Hugging Face Transformers Trainer."""

from dataclasses import dataclass
import json
import inspect
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

import evaluate


@dataclass
class FinetuneResult:
    model_dir: Path
    test_accuracy: float
    test_macro_f1: float


def finetune_distilbert_classifier(
    ds: DatasetDict,
    *,
    text_field: str,
    label_field: str = "label",
    model_name: str = "distilbert-base-uncased",
    max_length: int = 256,
    output_dir: Path = Path("artifacts/distilbert_pubmed_rct"),
    num_train_epochs: float = 2.0,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    logging_steps: int = 50,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    seed: int = 42,
    id2label: Optional[Dict[int, str]] = None,
    early_stopping_patience: int | None = 2,
    early_stopping_threshold: float = 0.0,
) -> Tuple[FinetuneResult, Dict[str, float]]:
    """
    Fine-tune a pretrained encoder for sequence classification using HF Trainer.
    Returns (summary_result, raw_metrics_dict).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        return tokenizer(
            batch[text_field],
            truncation=True,
            max_length=max_length,
        )

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=[c for c in ds["train"].column_names if c != label_field],
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Prefer dataset metadata when available (more robust than max(label)+1).
    num_labels: int
    label_feat = ds["train"].features.get(label_field)
    if label_feat is not None and hasattr(label_feat, "num_classes"):
        num_labels = int(getattr(label_feat, "num_classes"))
    else:
        num_labels = int(np.max(tokenized["train"][label_field])) + 1
    label2id = None
    if id2label:
        label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "macro_f1": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    use_fp16 = bool(torch.cuda.is_available())

    # Transformers occasionally renames TrainingArguments fields (e.g., evaluation_strategy -> eval_strategy).
    # Build kwargs in a forward-compatible way.
    sig_params = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "save_strategy": save_strategy,
        "logging_steps": logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": [],
        "seed": seed,
        "fp16": use_fp16,
    }
    if "eval_strategy" in sig_params:
        ta_kwargs["eval_strategy"] = eval_strategy
    else:
        ta_kwargs["evaluation_strategy"] = eval_strategy

    args = TrainingArguments(**ta_kwargs)

    callbacks = []
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_stopping_patience),
                early_stopping_threshold=float(early_stopping_threshold),
            )
        )

    # Transformers v5 removes `tokenizer=` from Trainer and replaces it with `processing_class=`.
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    metrics = trainer.evaluate(tokenized["test"])

    # Save training log history for curves/reporting
    try:
        (output_dir / "trainer_log_history.json").write_text(
            json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        # Don't fail training if logging can't be written.
        pass

    # Persist model & tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_acc = float(metrics.get("eval_accuracy", float("nan")))
    test_f1 = float(metrics.get("eval_macro_f1", float("nan")))

    return FinetuneResult(model_dir=output_dir, test_accuracy=test_acc, test_macro_f1=test_f1), {
        k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items()
    }


