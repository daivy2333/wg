# Tasks: m1-infrastructure

> 实现 M1 项目基础设施（任务 1.4-1.8）

## 1. 目录骨架（任务 1.4）

- [ ] 1.1 创建 `src/`、`src/data/`、`src/models/`、`src/evaluation/`、`src/utils/`、`notebooks/`、`outputs/`、`tests/` 目录
- [ ] 1.2 为所有 Python 包目录写入 `__init__.py`（src/、src/data/、src/models/、src/evaluation/、src/utils/、tests/）
- [ ] 1.3 验证 `python -c "import src.data"` 在 conda 环境无 ImportError

## 2. 依赖锁定（任务 1.5）

- [ ] 2.1 创建 `requirements.txt`，按 `openspec/specs/references/spec.md` 版本表写入全部依赖（含 pytest）
- [ ] 2.2 验证 `pip install -r requirements.txt` 在 conda 环境无版本冲突（dry-run 检查）

## 3. 数据加载模块（任务 1.6）

- [ ] 3.1 **TDD RED**：写 `tests/test_loader.py`，覆盖 Happy Path（125973, 43）/（22544, 43）+ Sad Path（FileNotFoundError）+ Edge（列数断言）
- [ ] 3.2 **Verify RED**：运行 `pytest tests/test_loader.py`，展示测试失败输出（loader 尚未实现）
- [ ] 3.3 实现 `src/data/loader.py`：定义 41 特征名常量 + `load_train()` / `load_test()` 函数（latin-1 编码，无 header）
- [ ] 3.4 **Verify GREEN**：运行 `pytest tests/test_loader.py`，展示全部测试通过输出
- [ ] 3.5 手工验证：`python -c "from src.data.loader import load_train, load_test; print(load_train().shape)"` 展示输出

## 4. 数据预处理模块（任务 1.7）

- [ ] 4.1 **TDD RED**：写 `tests/test_preprocessor.py`，覆盖缺失值检查 + 混合编码（OneHot + Label）+ StandardScaler + 难度列丢弃 + 二分类/多分类标签 + 预处理管线整合
- [ ] 4.2 **Verify RED**：运行 `pytest tests/test_preprocessor.py`，展示测试失败输出
- [ ] 4.3 实现 `src/data/preprocessor.py`：`check_missing()` + `encode_categorical()` + `standardize()` + `make_labels()` + `preprocess_pipeline()`
- [ ] 4.4 **Verify GREEN**：运行 `pytest tests/test_preprocessor.py`，展示全部测试通过输出
- [ ] 4.5 端到端验证：`from src.data.preprocessor import preprocess_pipeline; X, y = preprocess_pipeline(load_train(), task='binary'); print(X.shape, y.value_counts())` 展示输出

## 5. EDA Notebook（任务 1.8）

- [ ] 5.1 创建 `notebooks/01_data_exploration.ipynb`，使用 loader + preprocessor 完成基础 EDA
- [ ] 5.2 在 notebook 中调用 `load_train()` / `load_test()`，展示数据形状与标签分布
- [ ] 5.3 在 notebook 中绘制 4 大攻击类别（DoS/Probe/R2L/U2R）的频次柱状图
- [ ] 5.4 验证 notebook 可在 conda 环境从头运行（jupyter nbconvert --execute）

## 6. 验证与归档

- [ ] 6.1 运行 `pytest tests/ -v`，展示全部测试通过输出
- [ ] 6.2 运行 `python -c "import src.data; import src.models; import src.evaluation; import src.utils"`，展示无 ImportError
- [ ] 6.3 更新 `.claude/docs/tasks.md` 中 M1 任务 1.4-1.8 状态为 ✅ 已完成
- [ ] 6.4 更新 `.claude/docs/SNAPSHOT.md` 最近修改章节
- [ ] 6.5 运行 `openspec archive m1-infrastructure --yes` 归档变更