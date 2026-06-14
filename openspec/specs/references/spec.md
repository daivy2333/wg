# References Spec

> Version: 1.1.0 | Last Updated: 2026-06-14（同步 SNAPSHOT 依赖版本警告）

## Purpose

记录项目依赖和外部参考资源，确保依赖可追溯，资源可获取。

> ⚠️ **依赖版本同步问题（2026-06-14 发现）**：
> - `requirements.txt` 锁定的版本（如 pandas 3.0.2、torch 2.9.0、numpy 2.3.3、scikit-learn 1.8.0、matplotlib 3.10.8、seaborn 0.13.2）与实际 pip 安装版本（pandas 2.3.3、torch 2.11.0、numpy 2.2.6、scikit-learn ≥1.8、matplotlib 3.10.9、seaborn ≥0.13）不一致
> - M1 阶段使用 `pip install --dry-run` 验证只确认 "Requirement already satisfied"，未做版本一致性检查
> - **影响**: M1-M3 全部功能在当前实际环境下通过，但 `requirements.txt` 不能完全复现环境
> - **后续**: M4 启动前需重新对齐 `requirements.txt`（删旧文件重写 or pip freeze）

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

| 依赖名称 | 版本 | 官方链接 | 用途说明 |
|----------|------|----------|----------|
| Python（conda） | 3.13.5 | https://python.org | 主开发语言（`/home/daivy/miniconda3/bin/python`） |
| Pandas | 3.0.2 | https://pandas.pydata.org | CSV 读取、数据清洗、特征处理 |
| NumPy | 2.3.3 | https://numpy.org | 数值计算、数组操作 |
| Matplotlib | 3.10.8 | https://matplotlib.org | 实验图表绘制 |
| PyTorch | 2.9.0 | https://pytorch.org | 深度学习模型（CNN/LSTM 扩展） |
| scikit-learn | 1.8.0 | https://scikit-learn.org | 决策树/随机森林/MLP/评估指标 |
| Seaborn | 0.13.2 | https://seaborn.pydata.org | 统计可视化（热力图等） |

## 外部资源

| 资源名称 | 链接 | 关键内容摘要 |
|----------|------|-------------|
| NSL-KDD 官方论文 | IEEE CISDA 2009 | Tavallaee 等，数据集定义论文，开篇必引 |
| Kaggle NSL-KDD 数据集 | https://www.kaggle.com/datasets/hassan06/nslkdd | 训练集+测试集下载 |
| 深度学习 IDS 综述（2025） | AI-Enhanced IDS Using DL | CNN、RNN、LSTM 架构 |
| 随机森林+半监督学习（2025） | Semi-Supervised + RF | 准确率提升 8.92% |
| CNN-LSTM 混合方法 | Hybrid CNN-LSTM IDS | 结合 RFE 与决策树特征选择 |
