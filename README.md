# UNITER Base Code

这是一个面向“失落空间”课题的第一版 base code。当前版本只覆盖你已经确认的 MVP 范围：

- 只做“空间 + 意义”双模态
- 身份模态只预留骨架，默认关闭
- 不做遥感图
- 以区域为基本样本单位，而不是单张图像或单条文本

当前仓库的状态可以理解成：

- 主训练链路已经完整
- 推理、评估、导出、可视化和阈值标定已经接通
- `identity/IAI`、`lost_space` learned head、分割监督都已经留好接口
- 真实效果和论文终态仍然依赖你后续接入真实数据、标签和阈值校准

## 为什么第一版用这个架构

第一版不是复现原始 UNITER 预训练，而是采用 `UNITER-inspired Encode-Align-Decode` 思路：

- `Encode`
  - 空间分支：`SegFormer` 提取街景语义分割结果与空间向量
  - 文本分支：`MacBERT/BERT` 提取当前感知文本与历史文本的语义向量
  - 情感头：基于区域级当前/历史文本特征输出情感分类 logits
- `Align`
  - 将区域级图像向量和文本向量投影到同一个潜空间，使用对比学习进行对齐
- `Decode`
  - 根据分割结果构造 `IFI`
  - 优先根据当前/历史文本情感差异构造 `MDI`，关闭情感头时退回嵌入漂移代理

之所以选择 `SegFormer` 而不是单独从 `Swin Transformer backbone` 起步，是因为你的空间任务本质上更接近“街景语义分割 + 空间结构量化”，而 `SegFormer` 更适合直接作为工程基线。

## 当前目录

```text
.
├── configs/
│   └── base.toml
├── data/
│   └── templates/
│       └── region_manifest.example.jsonl
├── main.py
├── pyproject.toml
├── src/
│   └── uniter/
│       ├── cli.py
│       ├── config.py
│       ├── data/
│       ├── inference/
│       ├── metrics/
│       ├── models/
│       ├── reporting/
│       ├── training/
│       └── utils/
└── 需求/
```

## 数据接口

训练代码默认读取一个 `JSONL` manifest。每一行代表一个区域样本，格式如下：

```json
{
  "region_id": "wolongsi_front_square",
  "split": "train",
  "image_paths": [
    "street_view/wolongsi_front_square_001.jpg",
    "street_view/wolongsi_front_square_002.jpg"
  ],
  "segmentation_mask_paths": [null, null],
  "current_texts": [
    "门前空间比较空旷，但人停留的意愿不强。",
    "游客觉得这里交通噪声偏大，缺少历史场所感。"
  ],
  "historical_texts": [
    "旧志记载该处应具有较强的宗教礼仪与场所秩序感。"
  ],
  "identity_texts": [
    "宗教礼仪节点",
    "城市记忆界面"
  ],
  "metadata": {
    "city": "西安",
    "site_name": "卧龙寺"
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

- `image_paths`
  同一区域可放多张街景或实拍图，训练时会按区域聚合。
- `current_texts`
  当前现实感知文本，通常来自微博、点评、游客评论、访谈整理。
- `historical_texts`
  历史叙事文本。没有的话代码仍可跑，但 `MDI` 会是空值。
- `identity_texts`
  可选的身份语义文本接口，用于给后续 `IAI` 留骨架。当前默认不会进入主判定流程。
- `segmentation_mask_paths`
  可选的分割 mask 路径列表，长度需要和 `image_paths` 一致。接入后会启用分割监督接口。
- `targets.sentiment_label`
  当前版本默认按 3 类情感处理：`0=negative, 1=neutral, 2=positive`。没有标注时填 `null`，训练会自动跳过监督 loss。
- `targets.historical_sentiment_label`
  历史文本的可选情感标签。没有标注时填 `null`；有标注时会启用历史文本情感监督，并用于更稳定地估计 `MDI`。
- `targets.lost_space_label`
  可选的失落空间监督标签。支持二分类或 `none/light/moderate/severe` 四分类，取决于配置文件里的 `lost_space.num_classes`。
- `targets.iai`
  可选的 `IAI` 目标字段。当前不会参与默认训练，但会进入导出和评估摘要。
- `split`
  建议使用 `train / val / test`。

模板文件在 [data/templates/region_manifest.example.jsonl](/home/chelizi/project/uniter/data/templates/region_manifest.example.jsonl)。

如果你准备直接用目录扫描生成 manifest，推荐目录约定如下：

```text
data/
├── images/
│   ├── region_a/
│   │   ├── 001.jpg
│   │   └── 002.jpg
│   └── region_b/
├── current_texts/
│   ├── region_a.txt
│   └── region_b.txt
├── historical_texts/
│   ├── region_a.json
│   └── region_b.json
├── identity_texts/
│   ├── region_a.txt
│   └── region_b.txt
├── segmentation_masks/
│   ├── region_a/
│   │   ├── 001.png
│   │   └── 002.png
│   └── region_b/
└── metadata/
    ├── region_a.json
    └── region_b.json
```

其中：

- `images/<region_id>/` 下放该区域的多张图像
- `current_texts/<region_id>.txt` 可以是一行一条文本
- `historical_texts/<region_id>.json` 可以是 `{"texts": [...]}` 结构
- `segmentation_masks/<region_id>/` 中的文件名 stem 应和图像对应
- `metadata/<region_id>.json` 是任意 JSON object

## 配置

基础配置文件在 [configs/base.toml](/home/chelizi/project/uniter/configs/base.toml)。

当前关键配置包括：

- `data.manifest_path`
  你的真实区域样本 manifest 路径
- `spatial_model.model_name`
  默认是 `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`
- `text_model.model_name`
  默认是 `hfl/chinese-macbert-base`
- `sentiment.num_classes / sentiment.class_names`
  情感标签空间定义，默认三分类
- `losses.sentiment_weight`
  当前文本情感监督 loss 的权重；当 batch 没有情感标签时会自动跳过
- `losses.historical_sentiment_weight`
  历史文本情感监督 loss 的权重；只有存在 `historical_sentiment_label` 时才会生效
- `losses.lost_space_weight`
  `lost_space` 分类头监督 loss 的权重；需要同时打开 `lost_space.enabled` 并提供标签
- `losses.segmentation_weight`
  分割监督 loss 的权重；需要提供 `segmentation_mask_paths`
- `training.resume_from`
  从历史 checkpoint 恢复训练
- `training.save_best / training.monitor_metric / training.early_stopping_patience`
  best checkpoint、监控指标与早停策略
- `inference.require_checkpoint`
  默认推理是否强制要求 checkpoint，当前默认开启
- `metrics.ifi_target_profile`
  IFI 的目标空间轮廓，后续应基于你的样本统计重新校准
- `judgement.*`
  规则版失落空间判定阈值。当前默认使用 `IFI + MDI` 作为主指标，`alignment_gap` 作为辅助升级指标
- `judgement_fusion.*`
  最终判定融合层配置。代码已经支持规则判定和 `lost_space` 分类头做加权融合，也支持把 `IAI` 作为辅助信号接入；但只有在 `lost_space.enabled = true` 时 learned head 才会实际参与
- `calibration.*`
  阈值标定时使用的分位点和最小样本数
- 相对路径会按配置文件所在目录解析，所以示例配置里使用了 `../data/...` 这种写法

## 运行方式

先安装依赖：

```bash
uv sync
```

查看解析后的默认配置：

```bash
PYTHONPATH=src python3 -m uniter.cli describe-config
```

校验 manifest 结构：

```bash
PYTHONPATH=src python3 -m uniter.cli validate-manifest data/regions.jsonl --check-files
```

从目录结构生成 manifest：

```bash
PYTHONPATH=src python3 -m uniter.cli build-manifest \
  --output data/regions.jsonl \
  --image-dir data/images \
  --current-text-dir data/current_texts \
  --historical-text-dir data/historical_texts \
  --metadata-dir data/metadata
```

导出 manifest 摘要：

```bash
PYTHONPATH=src python3 -m uniter.cli summarize-manifest data/regions.jsonl
```

推荐的数据接入顺序：

1. 先准备目录结构或手写 `JSONL manifest`
2. 跑 `validate-manifest` 或 `build-manifest + summarize-manifest`
3. 再跑 `train`
4. 然后跑 `evaluate`
5. 最后导出 `export-region-metrics / export-visualizations`

开始训练：

```bash
PYTHONPATH=src python3 -m uniter.cli train --config configs/base.toml
```

从 checkpoint 恢复训练：

```bash
PYTHONPATH=src python3 -m uniter.cli train \
  --config configs/base.toml \
  --resume-from outputs/space_meaning_v1/checkpoints/epoch_005.pt
```

也可以直接运行：

```bash
python3 main.py train --config configs/base.toml
```

导出 split 级评估汇总：

```bash
PYTHONPATH=src python3 -m uniter.cli evaluate \
  --config configs/base.toml \
  --checkpoint outputs/space_meaning_v1/checkpoints/epoch_001.pt \
  --split val
```

默认会导出到 `outputs/.../evaluations/evaluation_<split>.json`，包含：

- `IFI / MDI / IAI / alignment_gap` 的数值汇总
- `IFI` 各组分偏差汇总
- 当前/历史情感分类指标
- `lost_space` 的 rule / model / final 三套指标
- 分割 `pixel_accuracy / mIoU / Dice / per-class IoU`

如果你只是调试随机初始化模型，需要额外显式传 `--allow-random-init`。
正式实验不建议这样做。

标定规则阈值：

```bash
PYTHONPATH=src python3 -m uniter.cli calibrate-thresholds \
  --config configs/base.toml \
  --checkpoint outputs/space_meaning_v1/checkpoints/epoch_001.pt \
  --split train
```

默认会导出到 `outputs/.../calibration/thresholds_<split>.json`，里面是按当前 split 分位点拟合出的 `IFI / MDI / alignment_gap` 轻中重阈值。

导出区域指标表：

```bash
PYTHONPATH=src python3 -m uniter.cli export-region-metrics \
  --config configs/base.toml \
  --checkpoint outputs/space_meaning_v1/checkpoints/epoch_001.pt \
  --split all
```

默认会导出到 `outputs/.../exports/region_metrics_all.csv`，包含每个区域的：

- `IFI / MDI / IAI / alignment_gap`
- `IFI` 组分 JSON 和 top groups
- 当前/历史情感 logits 与预测
- `lost_space` 的 rule level、model logits/pred、final fusion result
- `decision_summary`
- `mdi_source` 和身份可用性标记

导出可视化报告：

```bash
PYTHONPATH=src python3 -m uniter.cli export-visualizations \
  --config configs/base.toml \
  --checkpoint outputs/space_meaning_v1/checkpoints/epoch_001.pt \
  --split all
```

默认会导出到 `outputs/.../visualizations/<split>/`，其中包括：

- 每个区域一个 `report.md`
- 每张输入图像对应的语义分割叠图 `overlay_*.png`
- 一个总索引 `index.md`
- 如果训练 summary 存在，则额外生成 `training_curves.svg`

聚合多个实验输出目录：

```bash
PYTHONPATH=src python3 -m uniter.cli summarize-experiments \
  --root outputs \
  --output outputs/experiment_summary.json
```

## 当前实现包含什么

- 区域级样本 schema 与 manifest 校验
- 多图像、多文本的 batch 组装器
- `SegFormer` 空间分支
- `BERT/MacBERT` 文本分支
- 区域级情感分类头
- 区域级历史文本情感分类头（与当前文本共享标签空间）
- 可选 `lost_space` 分类头，支持二分类和四分类
- 区域级投影与图文对比学习
- 可选当前/历史情感监督 loss
- 可选分割监督接口与 segmentation loss
- 基于分割类别占比的 `IFI` 计算骨架
- 基于当前/历史文本情感差异的 `MDI` 计算，必要时退回嵌入漂移
- `identity_texts` 与 `IAI` 可选接口，默认不进入主训练监督
- 训练循环、checkpoint、history、best checkpoint、resume、early stopping
- 目录扫描式 manifest 构建与 manifest 摘要导出
- 区域级指标 CSV 导出
- split 级 evaluation JSON 导出
- 规则判定 + learned head 的最终融合判定层
  默认配置下仍以规则判定为主；只有打开 `lost_space.enabled` 后 learned head 才会进入最终融合
- `lost_space` 多分类评估、混淆矩阵和 macro-F1
- 分割 `mIoU / Dice / per-class IoU` 评估
- 基于分位点的阈值标定导出
- 区域 markdown 报告、分割叠图与训练曲线 SVG
- 多实验输出目录聚合摘要

## 当前实现不包含什么

- 遥感图分支
- 本地专用分割标签体系
- 本地情感微调数据
- 真实身份模态数据与 `IAI` 校准逻辑
- 基于真实遗产样本重新校准的论文阈值
- 最终论文使用的人工复核判定规则
- 真实数据驱动的 `identity/IAI` 终态监督

这些部分应该在你真实数据逐步到位后继续往里填。

## 你后面接数据时优先做什么

1. 先把所有研究区域整理成 `region_id`
2. 每个 `region_id` 收集多张街景/实拍图
3. 每个 `region_id` 收集当前文本
4. 如果要算完整 `MDI`，同步补历史文本
5. 生成 `data/regions.jsonl`
6. 先跑 `validate-manifest`
7. 再跑训练

## 备注

当前代码优先保证：

- 目录清晰
- 接口稳定
- 真实数据可直接接入
- 后续能继续扩展到身份模态
- 在没有监督信号时默认冻结预训练编码器，避免第一版训练把分割和文本模型训坏

你后面把第一批真实数据拿到后，建议优先补两件事：

- 本地街景小样本人工标注，用于检验分割结果是否真的适合遗产场景
- 本地评论/微博和历史文本小样本情感标注。这样当前/历史情感头和 `MDI` 都会更稳定。

当前版本里，`MDI` 会优先根据当前/历史文本的情感预测差异计算；如果显式关闭情感头，才会退回“当前文本嵌入 vs 历史文本嵌入”的代理漂移值。历史文本情感监督是可选的，你可以先只标注当前评论，再逐步补历史文本标签。

当前最终判定的默认思路是：

- 先用 `IFI + MDI (+ alignment_gap)` 生成规则判定
- 如果启用了 `lost_space` 分类头，则再和模型输出做加权融合
- 默认仍然以规则信号为主，模型信号需要达到最低置信度才会参与
- 如果后面打开 `judgement_fusion.use_iai = true`，则 `IAI` 也能作为辅助信号进入最终判定
- 阈值来自配置文件，也可以先用 `calibrate-thresholds` 基于现有样本自动拟合，再回填到配置里

输出目录里常见的结果文件包括：

- `outputs/<experiment>/checkpoints/`
- `outputs/<experiment>/summaries/history.json`
- `outputs/<experiment>/evaluations/evaluation_<split>.json`
- `outputs/<experiment>/exports/region_metrics_<split>.csv`
- `outputs/<experiment>/visualizations/<split>/`
- `outputs/<experiment>/calibration/thresholds_<split>.json`
- `outputs/experiment_summary.json`
