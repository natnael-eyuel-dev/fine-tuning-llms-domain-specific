from __future__ import annotations

"""Generate a filled evaluation report markdown from artifacts produced by the notebook/scripts."""

import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics_path = root / "artifacts" / "metrics.json"
    examples_path = root / "artifacts" / "examples.json"
    out_path = root / "reports" / "evaluation_report_filled.md"

    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}. Run the notebook first.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    examples = {}
    if examples_path.exists():
        examples = json.loads(examples_path.read_text(encoding="utf-8"))

    ds = metrics.get("dataset", {})
    baseline = metrics.get("baseline", {})
    finetune = metrics.get("finetune", {})
    split_sizes = ds.get("splits") or {}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def f(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "TBD"

    # A small sample of examples to paste into the report
    ex_rows = (examples.get("examples") or [])[:6]
    lines = []
    for e in ex_rows:
        txt = str(e.get("text", "")).replace("\n", " ")
        txt = txt[:220]
        true_lbl = e.get("true_label", e.get("true_id"))
        pred_lbl = e.get("pred_label", e.get("pred_id"))
        lines.append(f"- **correct={e.get('correct')}** true={true_lbl} pred={pred_lbl}: {txt}")
    ex_md = "\n".join(lines) or "- (Run notebook to generate `artifacts/examples.json`.)"

    out = f"""# Evaluation Report

Generated: **{now}**

## Dataset

- **Name/config**: `{ds.get('name', 'TBD')}` / `{ds.get('config', 'TBD')}`
- **Text field**: `{ds.get('text_field', 'TBD')}`
- **Split sizes**: {split_sizes if split_sizes else 'TBD (not recorded)'}

## Baseline vs Fine-tuned (Test)

| Model | Test Accuracy | Test Macro F1 |
|------|---------------:|--------------:|
| TF‑IDF + Logistic Regression | {f(baseline.get('test_accuracy'))} | {f(baseline.get('test_macro_f1'))} |
| PubMedBERT fine-tuned | {f(finetune.get('test_accuracy'))} | {f(finetune.get('test_macro_f1'))} |

## Notes

- Fine-tuning logs are saved at: `{finetune.get('model_dir', '')}/trainer_log_history.json` (if training ran).
- Raw metrics source: `{metrics_path}`
- If generated: `artifacts/training_curves.png`, `artifacts/confusion_matrix.png`, `artifacts/classification_report.json`, `artifacts/data_checks.json`

## Example predictions

{ex_md}
"""

    out_path.write_text(out, encoding="utf-8")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()


