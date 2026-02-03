from __future__ import annotations

"""Dataset loading and lightweight inspection helpers."""

from typing import Any, Dict, Tuple, cast

from datasets import Dataset, DatasetDict, load_dataset


def _infer_text_field(ds: Dataset) -> str:
    """
    Try to infer which column contains the input text.
    PubMed RCT commonly uses: "text" or "sentence".
    """
    candidates = ["text", "sentence", "abstract", "section_text", "content"]
    for c in candidates:
        if c in ds.column_names:
            return c
    # Fallback: first string column
    for name, feat in ds.features.items():
        if getattr(feat, "dtype", None) == "string":
            return name
    raise ValueError(f"Could not infer text field. Columns: {ds.column_names}")


def load_hf_classification_dataset(
    *,
    dataset_name: str,
    dataset_config: str | None = None,
    text_field: str | None = None,
    label_field: str = "label",
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_test_samples: int | None = None,
    seed: int = 42,
) -> Tuple[DatasetDict, str, Dict[int, str] | None]:
    """
    Loads a Hugging Face dataset (train/validation/test), optionally subsamples,
    and returns (dataset_dict, resolved_text_field, id2label).
    """
    ds = cast(DatasetDict, load_dataset(dataset_name, dataset_config))

    resolved_text = text_field or _infer_text_field(ds["train"])

    id2label: Dict[int, str] | None = None
    # Best-effort label names
    label_feat = ds["train"].features.get(label_field)
    if label_feat is not None and hasattr(label_feat, "names"):
        names = getattr(label_feat, "names")
        if isinstance(names, list):
            id2label = {i: n for i, n in enumerate(names)}

    def _subsample(split: str, n: int | None) -> Dataset:
        if n is None:
            return ds[split]
        return ds[split].shuffle(seed=seed).select(range(min(n, len(ds[split]))))

    # Some datasets use "validation", some use "val"
    eval_split = "validation" if "validation" in ds else ("val" if "val" in ds else None)
    if eval_split is None:
        raise ValueError(f"Dataset missing validation split. Splits: {list(ds.keys())}")

    ds = DatasetDict(
        train=_subsample("train", max_train_samples),
        validation=_subsample(eval_split, max_eval_samples),
        test=_subsample("test", max_test_samples),
    )

    # Ensure text field exists in all splits
    for split, split_ds in ds.items():
        if resolved_text not in split_ds.column_names:
            raise ValueError(
                f"Text field '{resolved_text}' not found in split '{split}'. "
                f"Columns: {split_ds.column_names}"
            )
        if label_field not in split_ds.column_names:
            raise ValueError(
                f"Label field '{label_field}' not found in split '{split}'. "
                f"Columns: {split_ds.column_names}"
            )

    return ds, resolved_text, id2label


def dataset_preview(ds: DatasetDict, text_field: str, label_field: str = "label", k: int = 3) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split in ["train", "validation", "test"]:
        rows = []
        for i in range(min(k, len(ds[split]))):
            rows.append({"text": ds[split][i][text_field], "label": ds[split][i][label_field]})
        out[split] = rows
    return out


