#!/usr/bin/env python3
"""Tiny ExtraTrees baseline for the easy-start structured 4-class table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


DATA_CSV = Path(__file__).resolve().parent / "structured_4class.csv"
SEEDS = [20260413, 20260414, 20260415]
ID_COLS = {"sample_id", "record_id", "label"}


def main() -> None:
    df = pd.read_csv(DATA_CSV)
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    x = df[feature_cols]
    y = df["label"]

    scores = []
    for seed in SEEDS:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=seed,
            stratify=y,
        )
        clf = ExtraTreesClassifier(
            n_estimators=200,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        scores.append(f1_score(y_test, pred, average="macro"))

    mean = pd.Series(scores).mean()
    std = pd.Series(scores).std(ddof=0)
    print(f"ExtraTrees macro-F1 over seeds {SEEDS}: {mean:.4f} +- {std:.4f}")


if __name__ == "__main__":
    main()
