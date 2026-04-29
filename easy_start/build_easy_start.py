#!/usr/bin/env python3
"""Build small, CSV-only starter tables for structured-feature baselines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent

FEATURES_CSV = REPO_ROOT / "tabular" / "structured_signal_features_full_v1.csv"
SAMPLES_CSV = REPO_ROOT / "metadata" / "samples.csv"

ID_COLS = ["sample_id", "record_id"]
STARTER_CLASSES = ["normal", "wss", "attenuation", "edfa"]
CONDITION_COLS = [
    "distance_km",
    "modulation",
    "modulation_order",
    "baud_gbaud",
    "lp_id",
    "condition",
    "fault_family",
    "severity_value_db",
    "wss_setting",
    "wss_offset_ghz",
    "wss_bandwidth_ghz",
    "repeat_idx",
]


def label5(row: pd.Series) -> str:
    """Split WSS into frequency-offset and filter-bandwidth examples."""
    if row["main_label"] != "wss":
        return row["main_label"]
    if pd.notna(row["wss_offset_ghz"]):
        return "wss_offset"
    if pd.notna(row["wss_bandwidth_ghz"]):
        return "wss_bandwidth"
    raise ValueError(f"WSS row has neither offset nor bandwidth setting: {row['sample_id']}")


def main() -> None:
    samples = pd.read_csv(SAMPLES_CSV)
    features = pd.read_csv(FEATURES_CSV)

    feature_cols = [c for c in features.columns if c not in ID_COLS]
    if len(feature_cols) != 34:
        raise ValueError(f"Expected 34 structured feature columns, found {len(feature_cols)}")

    sample_cols = ID_COLS + ["main_label", "sub_label"] + CONDITION_COLS
    merged = features.merge(samples[sample_cols], on=ID_COLS, how="inner", validate="one_to_one")
    if len(merged) != len(features):
        raise ValueError(f"Expected {len(features)} merged rows, found {len(merged)}")

    starter = merged[merged["main_label"].isin(STARTER_CLASSES)].copy()
    starter["label4"] = starter["main_label"]
    starter["label5"] = starter.apply(label5, axis=1)

    conditions = samples[ID_COLS + CONDITION_COLS + ["main_label", "sub_label"]].copy()
    conditions["label4"] = conditions["main_label"].where(conditions["main_label"].isin(STARTER_CLASSES))
    conditions["label5"] = conditions.apply(
        lambda row: label5(row) if row["main_label"] in STARTER_CLASSES else pd.NA,
        axis=1,
    )

    four_class_cols = ID_COLS + feature_cols + ["label"]
    five_class_cols = ID_COLS + feature_cols + ["label"]
    condition_cols = (
        ["sample_id", "record_id"]
        + feature_cols
        + CONDITION_COLS
        + ["sub_label", "label4", "label5"]
    )

    structured_4class = starter.assign(label=starter["label4"])[four_class_cols]
    structured_5class = starter.assign(label=starter["label5"])[five_class_cols]
    structured_conditions = starter[condition_cols]

    conditions.to_csv(OUT_DIR / "sample_conditions.csv", index=False)
    structured_4class.to_csv(OUT_DIR / "structured_4class.csv", index=False)
    structured_5class.to_csv(OUT_DIR / "structured_5class.csv", index=False)
    structured_conditions.to_csv(OUT_DIR / "structured_with_conditions.csv", index=False)

    print(f"Wrote sample_conditions.csv: {len(conditions)} rows")
    print(f"Wrote structured_4class.csv: {len(structured_4class)} rows")
    print(f"Wrote structured_5class.csv: {len(structured_5class)} rows")
    print(f"Wrote structured_with_conditions.csv: {len(structured_conditions)} rows")


if __name__ == "__main__":
    main()
