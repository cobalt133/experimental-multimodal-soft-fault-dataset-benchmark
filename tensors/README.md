Tensor modalities:

- `time_series_full_v1`: `3463` samples
- `iq_ds`: `(N, 4, 2048)`
- `amp_seq`: `(N, 2, 2048)`
- `phase_seq`: `(N, 2, 2048)`
- `psd_seq`: `(N, 3, 512)`

Tensor shard:

- `shard_0001.h5` is stored here as three split parts:
  - `shard_0001.h5.part-00`
  - `shard_0001.h5.part-01`
  - `shard_0001.h5.part-02`

Rebuild the HDF5 shard after cloning:

```bash
bash scripts/rebuild_tensor_shard.sh
```

The checksum files in this directory are used by the rebuild script for verification.
