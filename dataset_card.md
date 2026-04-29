# Dataset Card

## Overview

- Full indexed captures: `3463`
- Full structured-signal feature exports: `3463`
- Full time-series tensor exports: `3463`
- Export failures: `0`
- Sample IDs use the form `sample_000000`
- Record IDs use the form `record_000000`

## Modulation Counts

```json
{
  "16QAM": 1869,
  "4QAM": 1591,
  "64QAM": 3
}
```

## Group Counts

```json
{
  "wss": 1448,
  "edfa": 813,
  "normal": 682,
  "attenuation": 412,
  "b2b": 105,
  "reference_0km": 3
}
```

## LP Counts

```json
{
  "-1": 7,
  "1": 1248,
  "3": 1183,
  "5": 1025
}
```

## Distance Counts

```json
{
  "0": 108,
  "320": 524,
  "640": 1061,
  "960": 1060,
  "1280": 419,
  "1600": 96,
  "1920": 96,
  "2240": 99
}
```

## Dataset Tasks

- `full_index`: all indexed captures, metadata discovery layer.
- `structured_4class`: main benchmark subset for `normal / wss /
  attenuation / edfa` (`3355` samples).
- `structured_5class_wss_split`: same subset with WSS split into
  `wss_offset` and `wss_bandwidth`.
- `parameter_value_tasks`: within-family targets for attenuation dB, EDFA
  gain dB, WSS center-frequency offset, and WSS bandwidth setting.
- `full_group_map`: full label map for `normal`, `wss`,
  `attenuation`, `edfa`, `b2b`, and `reference_0km`.

## Easy-Start Condition Mapping

The `easy_start/` folder includes `sample_conditions.csv`, a lightweight
condition table for all `3463` samples. It can be joined to
`structured_4class.csv` or `structured_5class.csv` by `sample_id` and
`record_id`.

Important condition columns are:

- `distance_km`: transmission distance.
- `modulation`, `modulation_order`, `baud_gbaud`: signal format descriptors.
- `lp_id`: launch-power setting identifier.
- `fault_family`, `condition`, `main_label`, `sub_label`: label metadata.
- `severity_value_db`: attenuation or EDFA gain-deviation level when applicable.
- `wss_offset_ghz`, `wss_bandwidth_ghz`: WSS center-frequency offset and filter
  bandwidth settings when applicable.
