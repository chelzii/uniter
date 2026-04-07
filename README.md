# UNITER Base Code

面向“失落空间”课题的区域级多模态基线代码。当前仓库已经完成单区域 bootstrap 场景下的训练、阈值标定、评估、结果导出和可视化闭环，并把 `IFI / MDI / IAI` 三项指标纳入最终规则判定。

当前版本的工程定位：

- 以“区域”而不是单张图或单条文本为样本单位
- 支持空间图像、当前文本、历史文本、身份文本的联合建模
- 支持单区域案例分析，不把当前结果表述成多区域统计结论
- 支持从目录构建 manifest、从标注文件回填标签、统一导出实验产物

## 当前能力

- 区域级 manifest 校验、摘要导出、目录扫描构建
- CSV / JSON / JSONL 标注导入，支持回填 `lost_space_label`、当前/历史情感标签和 `IFI / MDI / IAI`
- `SegFormer` 空间分支 + `MacBERT/BERT` 文本分支
- `IFI / MDI / IAI` 三项指标计算与规则判定
- 单区域 bootstrap 保守判定与阈值校准
- 训练、评估、指标导出、单区域结果总表、Markdown 摘要、可视化导出
- 实验目录聚合与 `artifacts_index.json` 产物索引

## 仓库结构

```text
.
├── configs/
├── data/
├── datasets/
├── docs/
├── main.py
├── pyproject.toml
├── requirements.txt
├── src/uniter/
├── tests/
└── runs/
```

主要入口：

- `src/uniter/cli.py`：CLI 主入口
- `configs/base.yaml`：基础配置
- `configs/kaitong_west_lane_single_region.yaml`：单区域 bootstrap 基线配置
- `configs/kaitong_west_lane_single_region_rerun_optimized_20260406.yaml`：当前较新的复现实验配置
- `docs/base_technical_document.md`：技术说明

## 安装

推荐使用 `uv`：

```bash
uv sync
```

也可以直接安装最小依赖：

```bash
python3 -m pip install -r requirements.txt
```

开发和测试通常从仓库根目录执行：

```bash
PYTHONPATH=src python3 -m uniter.cli --help
```

或：

```bash
python3 main.py --help
```

## 数据接口

训练与推理默认读取 `JSONL manifest`。每行代表一个区域样本，最小结构如下：

```json
{
  "region_id": "wolongsi_front_square",
  "split": "train",
  "image_paths": ["street_view/001.jpg", "satellite/001.tif"],
  "segmentation_mask_paths": [null, null],
  "current_texts": ["门前空间比较空旷，但停留意愿不强。"],
  "historical_texts": ["旧志记载该处应具有较强场所秩序感。"],
  "identity_texts": ["宗教礼仪节点"],
  "metadata": {
    "city": "西安",
    "parent_region_id": "kaitong_west_lane",
    "bootstrap_view": "train_a"
  },
  "targets": {
    "lost_space_label": null,
    "sentiment_label": null,
    "historical_sentiment_label": null,
    "ifi": null,
    "mdi": null,
    "iai": null
  }
}
```

说明：

- `image_paths` 支持同一区域多张图像，街景图和卫星图统一走空间模态输入
- `historical_texts` 缺失时仍可运行，但 `MDI` 解释能力会下降
- `identity_texts` 默认进入 `IAI` 与导出链路，不要求始终参与监督训练
- `metadata.parent_region_id + metadata.bootstrap_view` 会触发单区域 bootstrap 上下文识别
- `targets` 中没有标注的字段可写 `null`

示例模板见 `data/templates/region_manifest.example.jsonl`。

## 推荐工作流

1. 检查或解析配置

```bash
PYTHONPATH=src python3 -m uniter.cli describe-config \
  configs/kaitong_west_lane_single_region.yaml
```

2. 从目录生成 manifest，或直接校验已有 manifest

```bash
PYTHONPATH=src python3 -m uniter.cli build-manifest \
  --output data/regions.jsonl \
  --image-dir datasets/data_workspace_cleaned/images \
  --current-text-dir datasets/data_workspace_cleaned/current_texts \
  --historical-text-dir datasets/data_workspace_cleaned/historical_texts \
  --identity-text-dir datasets/data_workspace_cleaned/identity_texts \
  --metadata-dir datasets/data_workspace_cleaned/metadata
```

```bash
PYTHONPATH=src python3 -m uniter.cli validate-manifest \
  data/regions.jsonl \
  --check-files
```

3. 导出 manifest 摘要

```bash
PYTHONPATH=src python3 -m uniter.cli summarize-manifest data/regions.jsonl
```

4. 把人工标注回填进 manifest

```bash
PYTHONPATH=src python3 -m uniter.cli import-annotations \
  --manifest data/regions.jsonl \
  --annotations-root datasets/annotations \
  --output data/regions.annotated.preview.jsonl
```

5. 训练

```bash
PYTHONPATH=src python3 -m uniter.cli train \
  --config configs/kaitong_west_lane_single_region.yaml
```

6. 标定阈值

```bash
PYTHONPATH=src python3 -m uniter.cli calibrate-thresholds \
  --config configs/kaitong_west_lane_single_region.yaml \
  --checkpoint runs/train/kaitong-west-lane-single-region-bootstrap/checkpoints/best.pt \
  --split train
```

7. 评估与导出

```bash
PYTHONPATH=src python3 -m uniter.cli evaluate \
  --config configs/kaitong_west_lane_single_region.yaml \
  --checkpoint runs/train/kaitong-west-lane-single-region-bootstrap/checkpoints/best.pt \
  --split all
```

```bash
PYTHONPATH=src python3 -m uniter.cli export-region-metrics \
  --config configs/kaitong_west_lane_single_region.yaml \
  --checkpoint runs/train/kaitong-west-lane-single-region-bootstrap/checkpoints/best.pt \
  --split all \
  --thresholds runs/train/kaitong-west-lane-single-region-bootstrap/calibration/thresholds_train.json
```

```bash
PYTHONPATH=src python3 -m uniter.cli export-region-results \
  --config configs/kaitong_west_lane_single_region.yaml \
  --checkpoint runs/train/kaitong-west-lane-single-region-bootstrap/checkpoints/best.pt \
  --split all
```

```bash
PYTHONPATH=src python3 -m uniter.cli export-visualizations \
  --config configs/kaitong_west_lane_single_region.yaml \
  --checkpoint runs/train/kaitong-west-lane-single-region-bootstrap/checkpoints/best.pt \
  --split all
```

如果你要一次性跑完单区域复现闭环，可以直接执行：

```bash
PYTHONPATH=src python3 -m uniter.cli reproduce-single-region \
  --config configs/kaitong_west_lane_single_region_rerun_optimized_20260406.yaml \
  --split all
```

## CLI 命令

当前 CLI 支持以下子命令：

- `describe-config`：解析并打印最终配置
- `validate-manifest`：校验 manifest 结构，可选检查文件是否存在
- `build-manifest`：从目录结构构建区域级 manifest
- `summarize-manifest`：输出 manifest 摘要 JSON
- `import-annotations`：把 CSV / JSON / JSONL 标注回填进 manifest
- `train`：训练模型并写入实验目录
- `calibrate-thresholds`：导出规则阈值
- `evaluate`：导出 split 级评估摘要
- `export-region-metrics`：导出区域指标 CSV
- `export-region-results`：导出单区域案例分析总表和 Markdown 摘要
- `export-visualizations`：导出 region report、分割叠图和训练曲线
- `summarize-experiments`：聚合多个实验目录
- `reproduce-single-region`：顺序执行训练、阈值标定、评估、结果导出和可视化

## 输出目录

所有训练和推理相关产物统一落在：

```text
runs/train/<experiment_name>/
├── artifacts_index.json
├── best_checkpoint.json
├── resolved_config.yaml
├── run.log
├── training_state.json
├── calibration/
├── checkpoints/
├── evaluations/
├── exports/
├── summaries/
└── visualizations/
```

其中常用文件包括：

- `summaries/data_summary.json`：训练前数据摘要
- `calibration/thresholds_train.json`：标定阈值
- `evaluations/evaluation_<split>.json`：评估摘要
- `exports/region_metrics_<split>.csv`：区域级指标表
- `exports/region_results_<split>.csv`：单区域结果总表
- `exports/region_results_<split>.md`：面向案例分析的 Markdown 摘要
- `artifacts_index.json`：当前实验目录的产物索引

## 当前内置数据与配置

仓库当前已经包含：

- `data/regions.jsonl`：当前训练入口 manifest
- `data/regions.single_region.full.jsonl`：单区域全量清单
- `data/regions.annotated.preview.jsonl`：标注回填预览结果
- `datasets/data_workspace_cleaned/`：清洗后工作区
- `datasets/annotations/`：人工标注目录
- `configs/kaitong_west_lane_single_region_rerun_optimized_20260406.yaml`：较新的单区域复现实验配置
- `runs/train/kaitong-west-lane-single-region-rerun-optimized-20260406/`：最近一轮完整实验输出

## 当前边界

当前版本明确不覆盖以下内容：

- 多区域正式统计推断
- 独立遥感分支
- 基于大规模人工 mask 的分割监督训练
- 新增模态，如声音、点云等
- 论文终态阈值和终态人工复核规则

## 文档

- 总体说明见 `README.md`
- 技术细节见 `docs/base_technical_document.md`
- 当前待办和完成情况见 `TODO.md`

## 测试

当前测试命令：

```bash
PYTHONPATH=src pytest -q
```

最近一次本地校验结果：

```text
82 passed, 2 warnings
```
