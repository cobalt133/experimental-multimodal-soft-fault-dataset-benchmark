# Easy Start

This folder contains small CSV entrypoints for users who want a quick
structured-feature experiment without loading tensors.

Build the starter tables:

```bash
python easy_start/build_easy_start.py
```

Then run the tiny structured-feature baseline:

```bash
python easy_start/quick_start_structured.py
```

## Outputs

- `sample_conditions.csv`: one row per sample (`3463` rows). Use
  `sample_id` and `record_id` to map any easy-start row to its experimental
  configuration, including transmission distance (`distance_km`), modulation,
  LP setting, fault family, attenuation/EDFA dB value, WSS center-frequency
  offset, and WSS bandwidth.
- `structured_4class.csv`: `sample_id`, `record_id`, 34 structured signal
  features, and final column `label`. Labels are `normal`, `wss`,
  `attenuation`, and `edfa`.
- `structured_5class.csv`: the same format, but WSS examples are split into
  `wss_offset` and `wss_bandwidth` using `wss_offset_ghz` and
  `wss_bandwidth_ghz`.
- `structured_with_conditions.csv`: `sample_id`, `record_id`, the 34 features,
  common experimental condition columns, `sub_label`, `label4`, and `label5`.

The quick-start script trains an `ExtraTreesClassifier` on
`structured_4class.csv` with seeds `20260413`, `20260414`, and `20260415`, then
prints mean `+-` standard deviation macro-F1.

## Match Features to Conditions

The compact feature tables keep the final `label` column simple. To attach the
experimental condition columns, join with `sample_conditions.csv`:

```python
import pandas as pd

features = pd.read_csv("easy_start/structured_4class.csv")
conditions = pd.read_csv("easy_start/sample_conditions.csv")

data = features.merge(
    conditions,
    on=["sample_id", "record_id"],
    how="left",
    validate="one_to_one",
)

print(data[["sample_id", "label", "distance_km", "modulation", "lp_id"]].head())
```
