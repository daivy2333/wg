# Architecture Spec

> Version: 1.0.0 | Last Updated: 2026-06-14

## Purpose

定义项目的架构决策和设计原则，指导开发过程中的技术选型和系统设计。

## Requirements

### Requirement: 架构决策记录

所有重要的架构决策 SHALL 以 ADR（Architecture Decision Record）形式记录，包含决策内容、原因、影响和替代方案。

#### Scenario: 记录新决策

- **WHEN** 开发者做出影响系统架构的决策（如选择模型、设计模块边界、确定数据处理流水线）
- **THEN** 必须创建新的 ADR 条目，包含：决策标题、决策内容、决策原因、影响范围、替代方案

#### Scenario: 查询已有决策

- **WHEN** 开发者需要了解某个架构选择的原因
- **THEN** 可以通过 grep 搜索 decision 标题或关键词快速定位相关 ADR

### Requirement: 架构原则遵循

所有架构设计 SHALL 遵循项目定义的架构原则。

#### Scenario: 评估设计方案

- **WHEN** 开发者提出新的设计方案
- **THEN** 方案必须符合架构原则（如关注点分离、数据与模型解耦），不符合时需说明理由

### Requirement: 模块边界清晰

系统模块之间 SHALL 有清晰的边界和接口定义。

#### Scenario: 新增模块依赖

- **WHEN** 模块 A 需要依赖模块 B
- **THEN** 必须通过明确定义的接口交互，禁止直接访问内部实现

### Requirement: 三人分工架构

项目 SHALL 按三人分工组织代码模块，确保各同学的工作可独立验证和合并。

#### Scenario: 组织代码结构

- **WHEN** 开发者创建新的代码文件或模块
- **THEN** 必须归属于 A同学（数据方向）、B同学（模型方向）或 C同学（对比分析）的职责范围，并在文件头注释中标注负责人

## ADR Records

### ADR-001: 技术栈选型

- **决策**: 使用 Python + Pandas/NumPy + scikit-learn + PyTorch 技术栈
- **原因**: 纯代码实现，本地环境即可完成；scikit-learn 适合传统 ML 模型，PyTorch 适合深度学习扩展
- **影响**: 所有代码基于 Python 生态，依赖 conda 环境管理
- **替代方案**: TensorFlow（已选择 PyTorch，更灵活）

### ADR-002: 数据集选择

- **决策**: 使用 NSL-KDD 数据集
- **原因**: KDD99 的改进版，去除冗余记录，是入侵检测领域标准基准
- **影响**: 41 特征，二分类/多分类任务，数据已下载在 dataset/archive.zip
- **替代方案**: KDD99（已被 NSL-KDD 取代）、CICIDS2017（与本项目方向不符）

### ADR-003: 项目分工模式

- **决策**: 三人分工 — A（数据）、B（模型）、C（对比分析）
- **原因**: 任务书要求明确分工，便于成果分配
- **影响**: 代码结构需按职责划分，milestone 设计需体现三人独立工作
- **替代方案**: 按功能模块分工（不利于成果分配）

### ADR-004: EDA 探索统一使用 .py 脚本而非 .ipynb

- **决策**: NSL-KDD 数据探索脚本采用 `.py` 文件（`notebooks/01_data_exploration.py`），不使用 Jupyter Notebook 格式
- **原因**:
  1. **版本控制友好**：`.py` 文件的 diff 清晰可读，便于 git review；`.ipynb` 包含 execution_count + base64 输出图片，diff 噪音大
  2. **CI/CD 友好**：可直接 `python scripts.py` 集成到自动化流程，无需 jupyter nbconvert 中间步骤
  3. **静态分析可用**：pyflakes / black / ruff 等工具可直接处理 `.py`，无需 jupyter 插件
  4. **图表输出明确**：脚本用 `matplotlib.use("Agg")` 后端 + `plt.savefig()` 保存 PNG，避免 nb 内嵌大图
  5. **教学与维护成本**：三人项目中小组成员不熟悉 jupyter 时，`.py` 可在任何 Python IDE 直接阅读
- **影响**:
  - `notebooks/` 目录保留作为 EDA 脚本容器，但内容为 `.py`
  - EDA 流程可通过 `python notebooks/01_data_exploration.py` 一键重跑
  - `docs/eda_report.md` 配套引用改为 `.py` 路径
  - OpenSpec `eda-reporting` spec 的 R11（"可从头运行"）改用 `python` 而非 `jupyter nbconvert` 验证
- **替代方案**:
  - Jupyter Notebook（`.ipynb`）：交互式更强，但 diff/版本控制/CI 体验差，已被本项目放弃
  - Streamlit / Dash 仪表盘：交互可视化更好，但引入 Web 框架，超出 M2 范围
  - Quarto / R Markdown：多语言支持好，但 Python 生态首选仍是 `.py`
