# Task Definitions

## Full Index

All `3463` indexed captures with metadata, structured features, and tensor rows.

## Four-Class Diagnosis

Main structured-feature diagnosis task:

- samples: `3355`
- classes: `normal / wss / attenuation / edfa`
- starter CSV: `easy_start/structured_4class.csv`

## Five-Class WSS Split

Five-class task that keeps the same `3355`-sample subset but separates WSS into
two metadata-backed mechanisms:

- `wss_offset`: WSS center-frequency offset examples using `wss_offset_ghz`
- `wss_bandwidth`: WSS filter-bandwidth examples using `wss_bandwidth_ghz`
- other classes: `normal / attenuation / edfa`
- starter CSV: `easy_start/structured_5class.csv`

## Parameter-Value Tasks

Within-family parameter-value targets are available from
`easy_start/structured_with_conditions.csv`:

- attenuation severity: `severity_value_db` for `attenuation`
- EDFA gain setting: `severity_value_db` for `edfa`
- WSS center-frequency offset: `wss_offset_ghz`
- WSS bandwidth setting: `wss_bandwidth_ghz`
