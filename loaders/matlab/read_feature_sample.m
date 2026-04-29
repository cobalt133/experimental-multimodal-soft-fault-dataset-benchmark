function S = read_feature_sample(dataset_root, sample_id)
% Minimal MATLAB helper for the public benchmark.

samples_csv = fullfile(dataset_root, 'metadata', 'samples.csv');
feats_csv = fullfile(dataset_root, 'tabular', 'structured_signal_features_full_v1.csv');

T = readtable(samples_csv);
idx = strcmp(T.sample_id, sample_id);
if ~any(idx)
    error('Sample %s not found.', sample_id);
end

S.meta = T(idx, :);

F = readtable(feats_csv);
idxf = strcmp(F.sample_id, sample_id);
if any(idxf)
    S.structured = F(idxf, :);
else
    S.structured = table();
end

ts_h5 = fullfile(dataset_root, 'tensors', 'time_series_full_v1', 'shard_0001.h5');
if exist(ts_h5, 'file')
    ids = h5read(ts_h5, '/sample_id');
    ids = cellstr(char(ids.'));
    pos = find(strcmp(ids, sample_id), 1);
    if ~isempty(pos)
        S.iq_ds = h5read(ts_h5, '/iq_ds', [1 1 1 pos], [4 2048 1 1]);
        S.amp_seq = h5read(ts_h5, '/amp_seq', [1 1 pos], [2 2048 1]);
        S.phase_seq = h5read(ts_h5, '/phase_seq', [1 1 pos], [2 2048 1]);
        S.psd_seq = h5read(ts_h5, '/psd_seq', [1 1 pos], [3 512 1]);
    end
else
    ts_part = fullfile(dataset_root, 'tensors', 'time_series_full_v1', 'shard_0001.h5.part-00');
    if exist(ts_part, 'file')
        warning(['Tensor shard has not been rebuilt yet. ' ...
            'Run "bash scripts/rebuild_tensor_shard.sh" from the repository root first.']);
    end
end
end
