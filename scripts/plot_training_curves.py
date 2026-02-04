from __future__ import annotations

"""Plot training curves from Trainer log history into a single PNG under artifacts/."""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_history(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_series(history: List[Dict[str, Any]], key: str) -> List[tuple[float, float]]:
    points: List[tuple[float, float]] = []
    for row in history:
        if key not in row:
            continue
        step = row.get("step")
        if step is None:
            continue
        try:
            points.append((float(step), float(row[key])))
        except Exception:
            continue
    return points


def main() -> None:
    artifacts = ROOT / "artifacts"

    metrics_path = artifacts / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}. Run notebook/script first.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model_dir = Path(metrics.get("finetune", {}).get("model_dir", ""))
    if not model_dir:
        raise SystemExit("metrics.json missing finetune.model_dir")

    history_path = model_dir / "trainer_log_history.json"
    if not history_path.exists():
        raise SystemExit(f"Missing {history_path}. Re-run fine-tuning to generate logs.")

    history = _load_history(history_path)

    train_loss = _extract_series(history, "loss")
    eval_loss = _extract_series(history, "eval_loss")
    eval_acc = _extract_series(history, "eval_accuracy")
    eval_f1 = _extract_series(history, "eval_macro_f1")

    out_png = artifacts / "training_curves.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    def plot(ax, series, title):
        if not series:
            ax.set_title(title + " (no data)")
            ax.axis("off")
            return
        xs, ys = zip(*series)
        ax.plot(xs, ys)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)

    plot(axes[0], train_loss, "Train loss")
    plot(axes[1], eval_loss, "Eval loss")
    plot(axes[2], eval_acc, "Eval accuracy")
    plot(axes[3], eval_f1, "Eval macro F1")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print("Wrote:", out_png)


if __name__ == "__main__":
    main()


