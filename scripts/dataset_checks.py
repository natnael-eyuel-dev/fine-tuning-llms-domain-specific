from __future__ import annotations

"""Dataset sanity checks: label distribution, duplicates, and split overlap (leakage risk)."""

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TrainConfig
from src.data import load_hf_classification_dataset


def _hash_texts(texts: Iterable[str]) -> list[str]:
    out = []
    for t in texts:
        t = (t or "").strip()
        # Hashing is only for duplicate/overlap detection; cryptographic security is not required here.
        out.append(hashlib.md5(t.encode("utf-8")).hexdigest())
    return out


def main() -> None:
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

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

    def label_dist(split: str) -> Dict[str, Any]:
        labels = list(ds[split][cfg.label_field])
        c = Counter(labels)
        dist = {str(k): int(v) for k, v in sorted(c.items(), key=lambda kv: kv[1], reverse=True)}
        return {"num_examples": len(labels), "label_counts": dist}

    checks: Dict[str, Any] = {
        "dataset": {"name": cfg.dataset_name, "config": cfg.dataset_config, "text_field": text_field},
        "splits": {k: len(v) for k, v in ds.items()},
        "label_distributions": {split: label_dist(split) for split in ds.keys()},
        "id2label": id2label,
    }

    # Duplicate + overlap checks (text-hash based)
    hashes = {split: _hash_texts(ds[split][text_field]) for split in ds.keys()}
    dup_within = {}
    for split, hs in hashes.items():
        c = Counter(hs)
        dup_within[split] = int(sum(v - 1 for v in c.values() if v > 1))
    checks["duplicates_within_split"] = dup_within

    splits = list(hashes.keys())
    overlap = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            overlap[f"{a}__{b}"] = int(len(set(hashes[a]).intersection(set(hashes[b]))))
    checks["overlap_between_splits"] = overlap

    out_path = artifacts / "data_checks.json"
    out_path.write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()


