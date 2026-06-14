# project-skeleton Spec

> Version: 1.0.0 | Last Updated: 2026-06-14

## ADDED Requirements

### Requirement: 标准项目目录结构

项目 SHALL 包含以下目录结构（每个 Python 包目录含 `__init__.py`）：

```
wg/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── notebooks/
├── outputs/
├── tests/
│   ├── __init__.py
│   ├── test_loader.py
│   └── test_preprocessor.py
└── requirements.txt
```

#### Scenario: 目录结构完整创建

- **WHEN** 运行 `mkdir -p` 创建目录 + 写入 `__init__.py`
- **THEN** 上述所有路径可通过 `os.path.exists()` 验证存在

#### Scenario: src 包可导入

- **WHEN** 在 conda 环境中执行 `python -c "import src.data"`
- **THEN** 无 ImportError，模块可被引用

### Requirement: 依赖锁定

`requirements.txt` SHALL 锁定所有依赖版本，与 `openspec/specs/references/spec.md` 一致。

#### Scenario: 包含核心数据科学依赖

- **WHEN** 检查 `requirements.txt` 内容
- **THEN** 必须包含 pandas/numpy/scikit-learn/seaborn/matplotlib/torch/pytest 全部带 `==` 版本号

#### Scenario: 版本与 references.md 一致

- **WHEN** 对比 `requirements.txt` 与 `openspec/specs/references/spec.md` 的依赖表
- **THEN** 每个依赖的版本号完全匹配

#### Scenario: 可安装验证

- **WHEN** 运行 `pip install -r requirements.txt`
- **THEN** 全部依赖成功安装，无版本冲突