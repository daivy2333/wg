## Context

当前 wg 项目仅完成 OpenSpec + 数据集 + 文档体系的初始化（commit a8d17e0），缺少可直接运行的 Python 代码骨架。conda 环境已就绪（Python 3.13.5 + 全套依赖），数据集 KDDTrain+.txt (125,973 行) 和 KDDTest+.txt (22,544 行) 已解压到 `dataset/`。M1 是 M2-M6 的前置依赖，若不打通会导致后续 EDA、训练无干净数据可用。

**关键约束**：
- Claude Code Bash 工具子 shell 中 conda 不可用，必须用 `/home/daivy/miniconda3/bin/python` 直接运行（参考 learned 踩坑-002）
- conda 实际版本：Python 3.13.5 + pandas 3.0.2 + numpy 2.3.3 + sklearn 1.8.0 + torch 2.9.0
- KDD 数据文件无 header，需硬编码 41 列特征名（NSL-KDD 学术规范）

## Goals / Non-Goals

**Goals:**
- 打通 M1 五项产出（目录骨架、requirements.txt、loader、preprocessor、EDA notebook）
- 建立 M2/M3/M4 可复用的数据加载与预处理 API
- TDD Iron Law：loader/preprocessor 必须有 pytest 测试见证
- 41 列特征名硬编码为单一来源常量（NSL-KDD 官方论文）

**Non-Goals:**
- 不实现模型训练（M3/M4 范围）
- 不实现高级特征工程（feature selection 在 M2 范围）
- 不修改 CLAUDE.md / OpenSpec 主 specs / .claude/docs/
- 不实现 SMOTE / 样本均衡（M4 范围）

## Decisions

### 决策 1: 41 特征名硬编码在 `src/data/loader.py` 常量

- **原因**：NSL-KDD 41 特征是固定学术规范（KDD Cup 1999 + NSL-KDD 论文），无需外部化配置
- **替代方案对比**：
  - YAML 配置：增加间接层，41 个常量查询多一次 IO
  - CSV 文件：同上
- **影响**：特征名修改只需改 loader.py 单文件

### 决策 2: 分类编码混合策略

- **protocol_type** (3 种) + **flag** (11 种)：OneHotEncoder
- **service** (70+ 种)：LabelEncoder
- **原因**：service 高基数 OneHot 会导致维度爆炸（70+ 列 → 整体 100+ 列）
- **替代方案**：全部 OneHot（维度灾难）；全部 Label（隐含顺序偏差）
- **影响**：service 列在混合策略下保留为单列整数

### 决策 3: StandardScaler 而非 MinMaxScaler

- **原因**：NSL-KDD 特征含长尾分布（如 src_bytes、dst_bytes），StandardScaler 对异常值更鲁棒
- **替代方案**：MinMaxScaler（对异常值敏感）、RobustScaler（IQR，中位数，备选）
- **影响**：标准化后 mean ≈ 0, std ≈ 1

### 决策 4: 保留 KDDTrain+/KDDTest+ 原始划分

- **原因**：NSL-KDD 设计目的就是验证模型对**未见过的攻击类型**的泛化能力（test 集含 14 种训练集未出现的攻击）
- **替代方案**：随机 80/20 划分（破坏 NSL-KDD 设计意图）
- **影响**：test 集必须严格只用 KDDTest+.txt

### 决策 5: 丢弃 `difficulty` 列

- **原因**：difficulty 是 NSL-KDD 评分（0-21），表示分类难度，不是特征。学术论文普遍丢弃。
- **影响**：loader 返回 43 列（41 特征 + label + difficulty），preprocessor 后输出 42 列（41 特征 + label）

### 决策 6: pytest 而非 unittest

- **原因**：pytest 是 Python 主流测试框架，fixture 机制简洁，社区标准
- **影响**：`tests/` 目录用 pytest 风格，conftest.py 可选

### 决策 7: 模块边界（data 与 models/evaluation 解耦）

```
src/
├── data/
│   ├── loader.py        # 加载 KDDTrain+/KDDTest+
│   └── preprocessor.py  # 编码 + 标准化 + 划分
├── models/              # M3/M4 范围，M1 只建空目录 + __init__.py
├── evaluation/          # M5 范围，M1 只建空目录 + __init__.py
└── utils/               # 通用工具（暂空，M1 占位）
```

- **原因**：M1 只打通 data 模块；其他模块目录占位但内容为 M3/M4/M5 留白
- **影响**：M3/M4 直接 `from src.data import loader, preprocessor` 即可

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| 41 特征名硬编码错误 | 模型输入错位 | 用 NSL-KDD 官方论文 + KDD Cup 1999 数据集文档双源验证 |
| KDDTrain+.txt 文件编码（Latin-1 vs UTF-8） | 读取异常 | 用 `encoding='latin-1'`（KDD 学术标准） |
| service 高基数 LabelEncoder 引入隐含顺序偏差 | 模型误读 | M2 阶段考虑 target encoding / frequency encoding（M1 范围内不处理） |
| OneHot 后维度膨胀（feature 数 41 → 80+） | 模型训练慢 | M1 不优化，M3 训练时观察 |
| Claude Code Bash 子 shell 无法激活 conda | 测试运行失败 | 测试用 `/home/daivy/miniconda3/bin/python -m pytest` 显式调用 |
| difficulty 列丢弃导致 M2 EDA 报告无法引用难度分布 | EDA 报告缺一项 | M1 不输出 difficulty 统计，M2 阶段若需要可单独加回 |

## Migration Plan

- **部署步骤**：M1 完成后，M2 启动前由 A 同学直接 `from src.data import load_train, preprocess_pipeline` 调用
- **回滚策略**：M1 仅新增文件/目录，无修改，回滚 = `git rm -r src/ notebooks/ requirements.txt`（依赖 git 跟踪新增文件）
- **依赖顺序**：M1 → M2 → M3/M4 → M5 → M6（M2 严格依赖 M1 的 loader + preprocessor）