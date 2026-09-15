# 实验多模态软故障数据集与基准

这是一个面向相干光传输系统软故障研究的独立开源数据集。仓库包含 `3463`
个接收端采集样本，每个样本提供 `34` 个结构化信号特征，并配套四种对齐的
时序张量模态。

最简单的使用方式是直接读取 CSV：前两列是公开样本 ID，随后是 34 个特征列，
最后一列是故障标签。

## 快速开始

四分类结构化特征表：

```text
data/structured_features_3355.csv
```

标签包括：

- `normal`
- `wss`
- `attenuation`
- `edfa`

运行一个最小示例：

```bash
pip install -r requirements.txt
python easy_start/quick_start_structured.py
```

其他入口表：

- `easy_start/sample_conditions.csv`：每个公开样本一行，用
  `sample_id` 和 `record_id` 对应距离、调制格式、LP 设置、故障类别、
  衰减/EDFA dB 数值、WSS 中心频率偏移和 WSS 带宽。
- `easy_start/structured_5class.csv`：在四分类基础上，将 WSS 拆分为
  `wss_offset` 和 `wss_bandwidth`。
- `easy_start/structured_with_conditions.csv`：额外包含距离、调制格式、
  LP、故障强度、WSS 中心频率偏移和 WSS 带宽等公开条件列，便于自定义分析。

如果希望把最简特征表和实验配置对应起来，可以直接按公开 ID 合并：

```python
import pandas as pd

features = pd.read_csv("data/structured_features_3355.csv")
conditions = pd.read_csv("easy_start/sample_conditions.csv")
data = features.merge(conditions, on=["sample_id", "record_id"], validate="one_to_one")
```

## 仓库内容

- `metadata/`：全部 `3463` 个样本的公开索引和实验条件元数据。
- `tabular/`：完整结构化特征表，每个样本 34 个建模特征。
- `tensors/`：时序张量索引和完整张量分片。
- `src/`：Python 预处理、模型、评估和绘图工具。
- `easy_start/`：最简 CSV 入口和结构化特征 baseline 示例。
- `feature_schema.yaml`、`label_schema.yaml`、`dataset_card.md`：数据说明文件。

## 张量数据

GitHub 对单个文件大小有限制，因此完整张量文件
`tensors/time_series_full_v1/shard_0001.h5` 以三个分片保存：

```text
tensors/time_series_full_v1/shard_0001.h5.part-00
tensors/time_series_full_v1/shard_0001.h5.part-01
tensors/time_series_full_v1/shard_0001.h5.part-02
```

克隆仓库后，如需使用张量数据，请运行：

```bash
bash scripts/rebuild_tensor_shard.sh
```

脚本会合并分片，并使用 SHA-256 校验文件完整性。
