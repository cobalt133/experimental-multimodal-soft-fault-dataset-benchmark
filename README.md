# Experimental Multimodal Soft-Fault Dataset Benchmark

[中文说明](README_zh.md)

This is the final five-class public benchmark for receiver-side soft-fault
diagnosis in coherent optical transmission systems. It provides 3355 primary
diagnosis records (from a 3463-record release), 34 structured descriptors and
aligned receiver representations. The public package contains derived data,
baseline models, fixed evaluation splits, metrics and figure-generation code.
Raw acquisition files and private DSP source are not included.

The diagnosis labels are `normal`, `voa_attenuation`, `edfa_gain_reduction`,
`wss_offset`, and `wss_bandwidth`. The nominal WSS bandwidth is 75 GHz;
narrowed settings are 70, 65, 60 and 55 GHz.

## Start With CSV

Use the five-class structured table:

```text
data/structured_features_3355.csv
```

The table contains capture identifiers, structured descriptors and the final
five-class label. A 2224-record low-BER analysis cohort is documented in the
metrics and data dictionaries.

Run a small structured-feature example:

```bash
pip install -r requirements.txt
python easy_start/quick_start_structured.py
```

Additional public materials:

- `data_dictionary/capture_index.csv`: maps each sample to its distance,
  modulation, LP setting, fault family, attenuation/EDFA dB value, WSS
  center-frequency offset, and WSS bandwidth.
- `data_dictionary/feature_schema.yaml`: public feature definitions.
- `splits/`: configuration-group and condition-holdout assignments.
- `metrics/`: saved benchmark summaries.
- `figures/main/`: manuscript figures.

Join easy-start features to conditions with the sample IDs:

```python
import pandas as pd

features = pd.read_csv("data/structured_features_3355.csv")
conditions = pd.read_csv("data_dictionary/capture_index.csv")
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
