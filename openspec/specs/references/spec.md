# References Spec

> Version: 1.2.0 | Last Updated: 2026-06-14（M4 新增 imbalanced-learn + PyTorch state_dict 文档 + M4 报告索引）

## Purpose

记录项目依赖和外部参考资源，确保依赖可追溯，资源可获取。

> ⚠️ **依赖版本同步问题（M1 2026-06-14 发现，M4 仍待修复）**：
> - `requirements.txt` 锁定的版本（如 pandas 3.0.2、torch 2.9.0、numpy 2.3.3、scikit-learn 1.8.0、matplotlib 3.10.8、seaborn 0.13.2）与实际 pip 安装版本（pandas 2.3.3、torch 2.9.0+cu128、numpy 2.2.6、scikit-learn ≥1.8、matplotlib 3.10.9、seaborn ≥0.13）不一致
> - M1 阶段使用 `pip install --dry-run` 验证只确认 "Requirement already satisfied"，未做版本一致性检查
> - **M4 新增**: imbalanced-learn 0.14.2（conda 实际安装）未在 requirements.txt 中记录
> - **影响**: M1-M4 全部功能在当前实际环境下通过，但 `requirements.txt` 不能完全复现环境
> - **后续（M5 前必做）**: 重新对齐 `requirements.txt`（`pip freeze > requirements.txt` 或手动重写），并补 imbalanced-learn

## Requirements

### Requirement: 依赖版本锁定

所有项目依赖 SHALL 记录版本信息，确保构建可重现。

#### Scenario: 添加新依赖

- **WHEN** 开发者引入新的外部依赖
- **THEN** 必须记录到 references/spec.md，包含：依赖名称、版本、官方链接、用途说明

#### Scenario: 更新依赖版本

- **WHEN** 开发者升级或降级依赖版本
- **THEN** 必须更新 references/spec.md 中的版本记录，标注更新原因

### Requirement: 外部资源记录

项目使用的外部资源和文档 SHALL 记录，方便查阅。

#### Scenario: 参考外部文档

- **WHEN** 开发者参考了重要的外部文档或资源
- **THEN** 必须记录到 references/spec.md，包含：资源名称、链接、关键内容摘要

### Requirement: 项目分析文档索引

深度分析文档 SHALL 建立索引，方便查找。

#### Scenario: 生成分析文档

- **WHEN** openspec-explorer 生成了项目分析文档
- **THEN** 必须在 references/spec.md 中注册索引条目，包含：主题、路径、内容概要

## 项目依赖

| 依赖名称 | 版本 | 官方链接 | 用途说明 | 引入里程碑 |
|----------|------|----------|----------|------------|
| Python（conda） | 3.13.5 | https://python.org | 主开发语言（`/home/daivy/miniconda3/bin/python`） | M1 |
| Pandas | 3.0.2 | https://pandas.pydata.org | CSV 读取、数据清洗、特征处理 | M1 |
| NumPy | 2.3.3 | https://numpy.org | 数值计算、数组操作 | M1 |
| Matplotlib | 3.10.8 | https://matplotlib.org | 实验图表绘制 | M1 |
| PyTorch | 2.9.0+cu128 | https://pytorch.org | 深度学习模型（MLP/CNN/LSTM） | M1（预先安装）→ M4 启用 |
| torchvision | 0.24.0+cu128 | https://pytorch.org/vision | 视觉工具（未直接使用，PyTorch 关联） | M4 |
| scikit-learn | 1.8.0 | https://scikit-learn.org | 决策树/随机森林/MLP/评估指标 | M1 |
| Seaborn | 0.13.2 | https://seaborn.pydata.org | 统计可视化（热力图等） | M1 |
| imbalanced-learn | 0.14.2 | https://imbalanced-learn.org | SMOTE 过采样（不均衡数据集） | M4（新增）|

> 注：M1 时期 requirements.txt 锁定 7 个依赖；M4 新增 imbalanced-learn（待 M5 前重新对齐 requirements.txt）

## 外部资源

| 资源名称 | 链接 | 关键内容摘要 | 引入里程碑 |
|----------|------|-------------|------------|
| NSL-KDD 官方论文 | IEEE CISDA 2009 | Tavallaee 等，数据集定义论文，开篇必引 | M1 |
| Kaggle NSL-KDD 数据集 | https://www.kaggle.com/datasets/hassan06/nslkdd | 训练集+测试集下载 | M1 |
| 深度学习 IDS 综述（2025） | AI-Enhanced IDS Using DL | CNN、RNN、LSTM 架构 | M1 |
| 随机森林+半监督学习（2025） | Semi-Supervised + RF | 准确率提升 8.92% | M1 |
| CNN-LSTM 混合方法 | Hybrid CNN-LSTM IDS | 结合 RFE 与决策树特征选择 | M1 |
| PyTorch state_dict 教程 | https://pytorch.org/tutorials/beginner/saving_loading_models.html | 官方推荐的 save/load 方式 | M4 |
| SMOTE 原始论文 | https://arxiv.org/abs/1106.1813 | Chawla 等 2002，少数类过采样基础 | M4 |
| imbalanced-learn API | https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html | `SMOTE(sampling_strategy, k_neighbors)` 参数 | M4 |
| tabular CNN/LSTM 局限性讨论 | https://arxiv.org/abs/2110.01889 | Shwartz-Ziv & Armon 2021，Tabular DL 综述 | M4 |
| Open-set Recognition 综述 | https://arxiv.org/abs/2207.01830 | 处理训练未见类的标准方法 | M4（O-NN-02）|

## 训练报告索引

| 报告名称 | 路径 | 内容概要 | 里程碑 |
|----------|------|----------|--------|
| EDA 报告 | `docs/eda_report.md` | M2 数据探索全流程与结论 | M2 |
| DT/RF 训练报告 | `docs/model_report_dt_rf.md` | M3 DT/RF 10 章节 + 多分类局限诚实记录 | M3 |
| MLP/DL 训练报告 | `docs/model_report_mlp_dl.md` | M4 MLP/CNN/LSTM 13 章节 + SMOTE 副作用诚实记录 | M4 |
