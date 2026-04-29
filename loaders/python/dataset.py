from pathlib import Path
import pandas as pd
import h5py


class PublicBenchmark:
    def __init__(self, root):
        self.root = Path(root)

    def _time_series_h5_path(self):
        path = self.root / "tensors" / "time_series_full_v1" / "shard_0001.h5"
        if path.exists():
            return path

        part0 = path.with_name("shard_0001.h5.part-00")
        if part0.exists():
            raise FileNotFoundError(
                f"{path} is not present yet. "
                "The tensor shard is stored as split parts. "
                "Run `bash scripts/rebuild_tensor_shard.sh` from the repository root first."
            )

        raise FileNotFoundError(f"Tensor shard not found: {path}")

    def records(self):
        return pd.read_parquet(self.root / "metadata" / "records.parquet")

    def samples(self):
        return pd.read_parquet(self.root / "metadata" / "samples.parquet")

    def structured_features(self):
        return pd.read_parquet(self.root / "tabular" / "structured_signal_features_full_v1.parquet")

    def time_series_index(self):
        return pd.read_parquet(self.root / "tensors" / "time_series_full_v1_index.parquet")

    def load_time_series(self, index: int):
        with h5py.File(self._time_series_h5_path(), "r") as h:
            return {
                "sample_id": h["sample_id"][index].decode("utf-8"),
                "iq_ds": h["iq_ds"][index],
                "amp_seq": h["amp_seq"][index],
                "phase_seq": h["phase_seq"][index],
                "psd_seq": h["psd_seq"][index],
            }
