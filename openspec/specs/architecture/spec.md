# Architecture Spec

> Version: 1.5.0 | Last Updated: 2026-06-14（M7 新增 ADR-011）

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

### ADR-005: sklearn 模型持久化使用 joblib 而非 pickle

- **决策**: M3 阶段 sklearn 模型（DT/RF）持久化统一使用 `joblib.dump/load`（`src/models/persistence.py`），M2 数据持久化仍用 `pickle`（`src/data/persistence.py`）
- **原因**:
  1. **大数组效率**：joblib 内部对 numpy array 采用分块序列化+压缩，RF 多分类模型（97MB）保存/读取速度比 pickle 快 5-10x
  2. **官方推荐**：scikit-learn 文档明确建议用 joblib 持久化 sklearn 模型（`sklearn.tree`、`sklearn.ensemble` 等）
  3. **依赖免费**：joblib 是 scikit-learn 的传递依赖，安装 sklearn 时自动获得，无需额外 `pip install joblib`
  4. **职责分离**：模型（joblib）与数据/特征（pickle）分属不同模块，独立维护更清晰
- **影响**:
  - `src/models/persistence.py` 出现（M3 新增），与 M2 的 `src/data/persistence.py` 共存
  - `outputs/models/*.joblib` 4 个文件（M3 产出）
  - `outputs/processed/*.pkl` 8 个文件（M2 产出，沿用 pickle）
  - M4+ 神经网络模型（torch.save）独立处理，不与 joblib 混用
- **替代方案**:
  - `pickle`（M2 方案）：通用但大 numpy 数组效率低
  - `onnx` 跨框架：模型可移植但需要额外 onnx 依赖，且对 sklearn 转换支持不完整
  - `pmml` 标准格式：标准化导出但生态较弱，本项目不需要跨工具兼容

### ADR-006: PyTorch 模型持久化用 state_dict + 双后端共存

- **决策**:
  - M4 神经网络模型持久化统一使用 `torch.save(model.state_dict(), path)`，加载时 `load_torch_model(model_class, path, **kwargs)` 还原
  - `src/models/persistence.py` 同时存在 `save_model/load_model`（joblib 后端，sklearn 用）和 `save_torch_model/load_torch_model`（torch.save 后端，PyTorch 用），调用接口不混用
  - `save_best_nn_models(output_dir, ...)` 整合函数支持 6-9 个 .pt 文件批量保存（None 表示跳过）
- **原因**:
  1. **state_dict 优势**：
     - 体积更小（仅参数，不绑类）
     - 不绑定 Python 类路径（避免 pickle 风险）
     - 跨 PyTorch 版本兼容性更好
  2. **双后端职责分明**：
     - joblib 负责 sklearn（DT/RF/M3）
     - torch.save 负责 PyTorch（MLP/CNN/LSTM/M4）
     - 调用方按模型类型选对应 API，模块不混用避免依赖冲突
  3. **批量保存**：
     - `save_best_nn_models` 接受 6 个可选模型参数
     - 未训练的模型传 None 自动跳过 + 打印警告（不全失败）
- **影响**:
  - `src/models/persistence.py` 扩展（M3 基础上 + M4 函数）
  - `outputs/models/*.pt` 6 个文件（M4 产出）
  - 加载示例：
    ```python
    from src.models.persistence import load_torch_model
    from src.models.mlp import MLPClassifier
    model = load_torch_model(MLPClassifier, "outputs/models/mlp_binary_tuned.pt",
                              input_dim=20, output_dim=2)
    preds = model.predict(X_test)
    ```
- **替代方案**:
  - `torch.save(model, path)` 存整个对象：体积大、依赖类路径、跨版本易失败
  - ONNX 跨框架：可移植但本项目不需要，且 PyTorch → ONNX 转换偶发算子不支持
  - PyTorch Lightning checkpoint：超框架，超出 M4 范围

### ADR-007: PyTorch + WSL 必须四层防御

- **决策**:
  - 测试环境（`tests/conftest.py`）强制四层防御：
    1. `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1`（环境变量）
    2. `torch.set_num_threads(1)` + `torch.set_num_interop_threads(1)`（CPU 线程）
    3. `CUDA_VISIBLE_DEVICES=""`（测试禁用 GPU）
    4. autouse fixture 每个测试后 `gc.collect()` + `torch.cuda.empty_cache()`（防内存累积）
  - 生产脚本（`scripts/train_m4.py`）用智能公式：`N_THREADS = min(4, max(1, cpu_count // 8))`
  - 模型 `predict` / `predict_proba` 方法内部 `device = next(self.parameters()).device` + `.to(device)`（避免 device mismatch）
- **原因**:
  1. **WSL2 内存限制**（通常 8GB），pytest 收集 14 文件 × 130 测试，每次 import 累积（sklearn + torch + imblearn）
  2. **PyTorch 默认 `torch.set_num_threads = cpu_count`**（WSL2 = 32 核 = 32 线程）
  3. **CUDA 初始化在 WSL2 不稳定**：占用 300-500MB 显存 + 触发 NVIDIA 驱动调用，对小数据完全无必要反而 OOM
  4. **Python 模型对象驻留 heap**：每个测试创建新模型，不显式释放 → 内存累积 → exit 137 → WSL 断开
- **影响**:
  - `tests/conftest.py` 从 2 层防御（M3 时期）扩到 4 层（M4 时期）
  - `src/models/mlp.py` / `cnn.py` / `lstm.py` 的 `predict` 方法必须先取 device 再 .to(device)
  - 生产脚本不再用 -1（隐式多线程）而是智能公式
  - **性能影响**：测试环境牺牲 < 5s，生产保留 ~80% CPU 利用率
- **预防措施**:
  - 任何 PyTorch + WSL 项目测试 conftest 必须含四层防御
  - 模型 predict/proba 必须处理 device
  - 长时间训练用 `python -u`（unbuffered）便于实时监控
  - 全量测试 > 100 个时分批跑（3-4 批次，防累积 OOM）
- **关联**:
  - 踩坑-003（pytest + sklearn 触发 WSL 断开 — M3 时期）
  - 踩坑-004（PyTorch + WSL 三层防御 — M4 初次实施）
  - 踩坑-005（Python stdout 缓冲 — M4 误判"卡死"）
  - O-PYTORCH-WSL-01（optimization 记录）
  - 模式-005（PyTorch 三层线程防御）
- **替代方案**:
   - 用 Docker 容器跑测试：环境隔离但学习成本高
   - pytest-xdist 并行：WSL 下反而加剧资源竞争
   - 减少测试数量：违反 TDD 完整性原则

### ADR-008: M5 统一测试集重评估原则

- **决策**: M5 对比分析阶段所有模型必须在统一的独立测试集（`X_test.pkl` + `y_test.pkl`）上重新评估，不直接复用 M3/M4 报告的验证集指标
- **原因**:
  1. M4 报告指标基于训练集内部验证划分（`train_val_split`），与测试集分布不同（IID vs OOD），MLP f1 从 0.989→0.720（差距 27%）
  2. M3 报告 DT/RF 用测试集评估但无机器可读 JSON 输出
  3. 公平对比要求所有模型在相同数据上评估：相同特征（20 维 Top-K）、相同样本（22,544 条）、相同标签
  4. 模型加载需精确 kwargs（MODEL_CONFIGS 字典），避免 PyTorch 维度不匹配
- **影响**:
  - `scripts/evaluate_m5.py` 加载 10 个模型统一 predict，输出 `metrics_m5.json`
  - 论文所有指标引用 M5 的测试集值，M4 报告值标注为"训练参考值"
  - src/evaluation/ 模块提供模型无关的评估函数（纯 sklearn/numpy，不绑定模型对象）
  - 多分类评估使用 full_accuracy + known_class_accuracy 双数字（遵循 M4 `train_m4.py:96-130` 模式）
- **替代方案**:
  - 直接复用 M3/M4 报告指标：不可靠（验证集 vs 测试集混淆，指标虚高）
  - 每个模型单独评估脚本：代码重复，不一致

### ADR-009: 对比分析模块化架构（src/evaluation/ + scripts/）

- **决策**: 评估逻辑分为两层——`src/evaluation/`（纯函数模块，无文件 I/O）+ `scripts/evaluate_m5.py`（编排脚本，文件 I/O + 模型加载）
- **原因**:
  1. 指标计算（`metrics.py`）和可视化（`plot.py`）是纯函数，可测试、可复用、模型无关
  2. 编排脚本负责文件 I/O、模型加载、报告生成——遵循 M3 `train_m3.py` 编排风格
  3. 两个模块可并行开发（仅依赖 API 签名合同），加速开发 55%
  4. 评估函数不绑定具体模型对象（接受 `y_true/y_pred/y_prob` 数组），可用于任何模型
- **影响**:
  - `src/evaluation/metrics.py`：3 个函数（二分类/多分类/分类 F1），14 个测试
  - `src/evaluation/plot.py`：5 个函数（混淆矩阵/ROC/柱状图/特征重要度/DLvsML），Agg 后端 + dpi=80
  - 编排脚本内置 `compute_binary_metrics()` / `_get_probabilities()` 等 inline 函数（自包含，不依赖 src/evaluation 导入）
  - 图表编号 09-13（M2 占 01-08），输出到 `outputs/figures/`
- **替代方案**:
   - 单文件脚本（`train_m3.py` 风格）：简单但不可测试，函数不可复用
   - Notebook（`evaluate_m5.ipynb`）：交互式但不可自动化

### ADR-010: 论文 LaTeX 项目结构（ctex + 章节分离）

- **决策**: 论文使用 LaTeX + ctex 文档类（XeLaTeX 编译），章节按独立 .tex 文件组织，通过 `\input{}` 合并到主文件
- **原因**:
  1. ctex 原生支持 UTF-8 中文，无需 CJK 编码转换
  2. 章节独立文件便于三人并行撰写（A: Ch1+2, B: Ch3, C: Ch4+5），无内容依赖
  3. `\graphicspath{{../outputs/figures/}}` 统一图表路径，避免 15 张 PNG 重复复制
  4. 图表通过 `paper/figures/` 符号链接到 `outputs/figures/`，保持单一数据源
  5. BibTeX + `\bibliographystyle{unsrt}` 管理参考文献（9 条目）
- **影响**:
  - `paper/main.tex`：主文件含封面、摘要、目录、`\input` 6 章、参考文献
  - `paper/chapters/ch1-intro.tex` ~ `ch6-conclusion.tex`：6 个独立章节
  - `paper/refs.bib`：NSL-KDD, sklearn, PyTorch, SMOTE, 3 篇中文文献
  - `paper/slides_outline.md`：15 页答辩 PPT 提纲
  - `README.md`：252 行完整复现指南（5 步：EDA→M3→M4→M5→测试）
  - 论文总计 9,764 中文字符，引用 15 张图表，满足 ≥5,000 字要求
- **替代方案**:
  - Word 文档：所见即所得但版本控制不友好，图表管理繁琐
  - Overleaf 在线 LaTeX：协作方便但依赖网络
   - Markdown + Pandoc 转 LaTeX：转换过程可能丢失格式细节

### ADR-011: M7 论文迁移至 CJC 模板（课程作业适配，终稿）

- **决策**: 将 M6 论文从 `ctexart` + 双栏排版迁移至 CJC（《计算机学报》）LaTeX 模板 `cjc.cls`，针对课程作业场景做轻量适配，并完成终稿润色
- **原因**:
  1. CJC 模板提供标准化 `\classsetup{}` 双语元数据
  2. 模板内置 `background`/`appendix` 等期刊环境
  3. `cjc.bst` 顺序引用格式符合中文期刊规范
  4. 课程作业适配：去除 CLC/DOI/日期/通信作者/期刊顶栏/算法伪代码
  5. 补充 10 条深度学习入侵检测领域长篇文章引用（总 19 条）
- **影响**:
  - 新建 `paper/cjc-main.tex`（主文件），作者：刘卫/查恩鹏/陈安旭，青海大学
  - 图1-8/10/13-16 使用 `figure*` 双栏跨页，图9/11/12 使用 `figure` 单栏
  - 表4 使用 `table*` 双栏，其余表保持单栏
  - 所有图表引用硬编码（0 个 `\ref{}`），消除编译依赖
  - `paper/cjc.cls` 打 3 处补丁（注释 CLC/DOI、收稿日期、通信作者渲染行）
  - `paper/cjc-refs.bib`：19 条参考文献
  - 去除正文中 M5 等开发阶段标记、代码路径引用
- **替代方案**:
  - 继续用 `ctexart` 排版：排版自由但缺期刊标准格式
  - Word 模板：版本管理困难

