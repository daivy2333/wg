## Why

M3（传统 ML）和 M4（神经网络）各产出了 4 个模型的训练结果和评估指标，但指标分散在三份独立报告（`eda_report.md`、`model_report_dt_rf.md`、`model_report_mlp_dl.md`）中，评估数据不统一（M3 用测试集、M4 部分用验证集），缺乏横向对比基线。M5 需要在统一的测试集上重新评估全部 10 个模型，生成论文级别的对比图表和综合分析报告，为 M6 论文撰写提供完整的实验数据支撑。

## What Changes

- 新建 `src/evaluation/` 模块（`metrics.py` + `plot.py`），提供模型无关的评估指标计算与可视化函数
- 新建 `scripts/evaluate_m5.py` 编排脚本，统一加载 10 个模型并在测试集上重新评估
- 生成 7 个对比图表（混淆矩阵热力图、性能对比柱状图、ROC 曲线叠加图、特征重要度对比图等）
- 生成 `docs/comparison_report.md` 完整对比分析报告（7 章节）
- 生成 `outputs/metrics_m5.json` 汇总指标文件（机器可读）
- 生成 `outputs/label_id_to_name.json` 多分类标签映射文件
- **不修改** 任何 M3/M4 的现有训练代码、模型文件、报告文件

## Capabilities

### New Capabilities

- `comparison-evaluation`: 统一的模型评估框架，支持 sklearn 和 PyTorch 双后端，在测试集上计算二分类（accuracy/precision/recall/f1/auc）和多分类（full_accuracy/known_class_accuracy/f1_macro）指标
- `comparison-visualization`: 论文级别的对比可视化（混淆矩阵热力图、性能柱状图、ROC 曲线、特征重要度对比）
- `comparison-reporting`: 自动生成包含全部 7 个任务结论的综合对比分析报告

### Modified Capabilities

<!-- M5 仅新增能力，不修改现有 spec 的 requirements -->

## Impact

- **新增文件**: `src/evaluation/metrics.py`、`src/evaluation/plot.py`、`scripts/evaluate_m5.py`
- **新增产出**: `docs/comparison_report.md`、`outputs/metrics_m5.json`、`outputs/label_id_to_name.json`、`outputs/figures/09_*` ~ `outputs/figures/15_*`（7 张图表）
- **依赖**: `src/data/persistence.py`（数据加载）、`src/models/persistence.py`（模型加载）、`src/models/mlp.py`（MLPClassifier）、`src/models/cnn.py`（CNN1DClassifier）、`src/models/lstm.py`（LSTMClassifier）
- **不影响**: M1-M4 所有已有文件
