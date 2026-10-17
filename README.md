# Experimental multimodal soft-fault dataset benchmark

Start with `data/structured_features_3355.csv` (3355 captures, five diagnosis labels, 32 explicit model inputs). `structured_features_2224_low_ber.csv` is the capture-level low-BER subset. Do not infer model inputs by selecting all numeric metadata columns.

## Contents

- `data_dictionary/`: labels, capture/configuration identifiers, operating conditions, QoT eligibility, feature whitelist and array shapes.
- `data/`: clean primary and low-BER tables; multimodal index; existing public derived tensor parts. No raw oscilloscope files or private DSP code.
- `splits/`: exact train/test membership for each experiment and cohort. Configuration repetitions stay together except in the explicitly named capture-random comparison.
- `predictions/`, `metrics/`: saved held-out predictions, confusion counts, uncertainty summaries and fold eligibility.
- `src/models/`: four fixed reference models; `src/evaluation/`: strict replay and verification; `src/figures/`: figures from saved metrics.
- `scripts/`: command-line entry points; `figures/`: PDF outputs; `reproduction/`: integrity and validation reports.

`figures/main/` contains the figures used in the manuscript. `figures/supplementary/` contains individual extended figures. [Additional Material](docs/Additional_Material.pdf) collects classification and transfer results, condition-resolved BER, and representative constellations indexed by all 299 physical configurations. Its contents and configuration index link to the corresponding pages. Numerical results and plotting code are provided in the folders listed above.

The acquisition-date stress test changes launch power as well as date: the 990-capture group contains 989 captures at 5 dBm and one with unrecorded power; the 2365-capture group contains 1214 at 1 dBm and 1151 at 3 dBm. It does not isolate a date effect or measure transfer between independently identified acquisition sessions. Exact memberships and predictions are retained under `acquisition_day_proxy`.

## Quick start

Install `pip install -r requirements.txt` in an isolated Python environment. These numerical-library versions match the reference environment; randomized PCA can give slightly different predictions with other versions. On macOS, LightGBM additionally requires an OpenMP runtime (for example, libomp). From the repository root run:
`python scripts/quick_start.py`
`python scripts/run_baselines.py --cohort full --model lightgbm`
`python scripts/run_targeted_analysis.py --cohort full --experiment holdout_exact_cell`
`python scripts/plot_results.py`

Replay outputs are written to `reproduction/reruns/` and never overwrite the saved reported predictions. Complete-condition holdout uses its documented five-class support (1867 captures), which is a subset of the 3355 primary resource. Parameter-specific and condition-holdout experiments have their own eligibility lists; absent class support is not a zero score. Macro-F1 and confusion matrices must be computed from the same held-out predictions. Intervals condition on these fitted predictions and resample physical configurations.

`amp_seq` stores intensity. Tensor concatenation: `python scripts/rebuild_tensor_shard.py`. Read `data_dictionary/qot_definition.md` before using BER membership.
