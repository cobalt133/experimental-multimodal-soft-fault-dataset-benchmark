# Experimental Multimodal Soft-Fault Dataset Benchmark

[中文说明](README_zh.md)

This is a standalone public dataset for soft-fault studies in coherent optical
transmission systems. It contains `3463` receiver-side captures from controlled
experiments, with `34` structured signal features and four aligned time-series
tensor modalities for each sample.

The easiest entry point is a plain CSV table: sample identifiers first, feature
columns next, and the fault label last. Tensor data and reproducibility assets
are included for users who want to go beyond tabular baselines.

## Start With CSV

Use the four-class structured table:

```text
easy_start/structured_4class.csv
```

The columns are `sample_id`, `record_id`, `34` structured features, and final
column `label`. Labels are `normal`, `wss`, `attenuation`, and `edfa`.

Run a small structured-feature example:

```bash
pip install -r requirements.txt
python easy_start/quick_start_structured.py
```

Additional starter tables:

- `easy_start/sample_conditions.csv`: maps each sample to its distance,
  modulation, LP setting, fault family, attenuation/EDFA dB value, WSS
  center-frequency offset, and WSS bandwidth.
- `easy_start/structured_5class.csv`: splits WSS into `wss_offset` and
  `wss_bandwidth`.
- `easy_start/structured_with_conditions.csv`: adds distance, modulation, LP,
  severity, WSS center-frequency offset, and WSS bandwidth columns for custom
  analysis.

Join easy-start features to conditions with the sample IDs:

```python
import pandas as pd

features = pd.read_csv("easy_start/structured_4class.csv")
conditions = pd.read_csv("easy_start/sample_conditions.csv")
data = features.merge(conditions, on=["sample_id", "record_id"], validate="one_to_one")
```

## Repository Contents

- `metadata/`: sample and record indices for all `3463` captures
- `tabular/`: full structured feature exports (`34` modeling features per sample)
- `tensors/`: full tensor indices plus the complete time-series tensor shard
- `loaders/`: Python and MATLAB helpers for loading samples and features
- `easy_start/`: CSV-only entrypoint for the main structured tasks
- `feature_schema.yaml`, `label_schema.yaml`, and `dataset_card.md`: dataset documentation

## Tensor Shard

GitHub rejects individual files larger than `100 MB`.
The tensor file `tensors/time_series_full_v1/shard_0001.h5` is stored as three
split parts:

- `tensors/time_series_full_v1/shard_0001.h5.part-00`
- `tensors/time_series_full_v1/shard_0001.h5.part-01`
- `tensors/time_series_full_v1/shard_0001.h5.part-02`

Rebuild it after cloning:

```bash
bash scripts/rebuild_tensor_shard.sh
```

The script verifies the rebuilt file against the recorded SHA-256 checksum before keeping it.

## Python Loader

```python
from loaders.python.dataset import PublicBenchmark

ds = PublicBenchmark(".")
samples = ds.samples()
features = ds.structured_features()

# Run this once after clone if you need tensors:
# bash scripts/rebuild_tensor_shard.sh
example = ds.load_time_series(0)
```
