from pathlib import Path
import h5py
import pandas as pd


def load_table(root: str | Path, relative_path: str) -> pd.DataFrame:
    root = Path(root)
    path = root / relative_path
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def read_h5_dataset(root: str | Path, relative_path: str, key: str):
    root = Path(root)
    with h5py.File(root / relative_path, "r") as h:
        return h[key][:]
