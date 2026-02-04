from __future__ import annotations

"""Error analysis for the fine-tuned classifier: per-class metrics + confusion matrix + failures."""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    artifacts = ROOT / "artifacts"
    metrics_path = artifacts / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}. Run notebook/script first.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_dir = Path(metrics.get("finetune", {}).get("model_dir", ""))
    if not model_dir:
        raise SystemExit("metrics.json missing finetune.model_dir")

    # Load dataset the same way as training (best-effort; relies on HF dataset availability)
    from src.config import TrainConfig
    from src.data import load_hf_classification_dataset

    cfg = TrainConfig()
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

    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    failures: List[Dict[str, Any]] = []

    # Batch inference (simple)
    batch_size = 32
    texts = list(ds["test"][text_field])
    labels = list(ds["test"][cfg.label_field])

    for i in range(0, len(texts), batch_size):
        batch_text = texts[i : i + batch_size]
        enc = tok(batch_text, return_tensors="pt", truncation=True, max_length=cfg.max_length, padding=True)
        with torch.no_grad():
            out = mdl(**enc)
        preds = out.logits.argmax(dim=-1).cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(labels[i : i + batch_size])

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    labels_sorted = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()))
    target_names = None
    if id2label:
        target_names = [id2label[i] for i in labels_sorted]

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels_sorted,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    (artifacts / "classification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels_sorted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=target_names or labels_sorted,
        yticklabels=target_names or labels_sorted,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Fine-tuned Model)")
    plt.tight_layout()
    cm_png = artifacts / "confusion_matrix.png"
    plt.savefig(cm_png, dpi=160)
    plt.close()

    # Save a small set of failures
    def to_label(x: int) -> Any:
        return id2label.get(int(x)) if id2label else int(x)

    for idx, (t, p, text) in enumerate(zip(y_true_arr, y_pred_arr, texts)):
        if t == p:
            continue
        failures.append(
            {
                "index": idx,
                "true": int(t),
                "pred": int(p),
                "true_label": to_label(t),
                "pred_label": to_label(p),
                "text": str(text)[:500],
            }
        )
        if len(failures) >= 50:
            break

    (artifacts / "failures.json").write_text(
        json.dumps({"failures": failures}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Wrote:", artifacts / "classification_report.json")
    print("Wrote:", cm_png)
    print("Wrote:", artifacts / "failures.json")


if __name__ == "__main__":
    main()


