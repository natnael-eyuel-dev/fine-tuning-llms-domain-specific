from __future__ import annotations

"""Classical baseline models for fair comparison against fine-tuning."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class BaselineResult:
    accuracy: float
    macro_f1: float


def _get_xy(split: Dataset, *, text_field: str, label_field: str) -> Tuple[List[str], np.ndarray]:
    x = [t if isinstance(t, str) else str(t) for t in split[text_field]]
    y = np.asarray(split[label_field])
    return x, y


def train_tfidf_logreg_baseline(
    ds: DatasetDict,
    *,
    text_field: str,
    label_field: str = "label",
    max_features: int = 60_000,
    seed: int = 42,
) -> Tuple[BaselineResult, Dict[str, float]]:
    """
    Fast baseline for comparison: TF-IDF + multinomial logistic regression.
    Returns (test_metrics, all_metrics) where all_metrics includes val+test.
    """
    x_train, y_train = _get_xy(ds["train"], text_field=text_field, label_field=label_field)
    x_val, y_val = _get_xy(ds["validation"], text_field=text_field, label_field=label_field)
    x_test, y_test = _get_xy(ds["test"], text_field=text_field, label_field=label_field)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        lowercase=True,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    x_val_vec = vectorizer.transform(x_val)
    x_test_vec = vectorizer.transform(x_test)

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=None,
        verbose=0,
        random_state=seed,
    )
    clf.fit(x_train_vec, y_train)

    def _metrics(xv, yv) -> Dict[str, float]:
        pred = clf.predict(xv)
        return {
            "accuracy": float(accuracy_score(yv, pred)),
            "macro_f1": float(f1_score(yv, pred, average="macro")),
        }

    val_m = _metrics(x_val_vec, y_val)
    test_m = _metrics(x_test_vec, y_test)
    all_m = {f"val_{k}": v for k, v in val_m.items()} | {f"test_{k}": v for k, v in test_m.items()}

    return BaselineResult(accuracy=test_m["accuracy"], macro_f1=test_m["macro_f1"]), all_m


