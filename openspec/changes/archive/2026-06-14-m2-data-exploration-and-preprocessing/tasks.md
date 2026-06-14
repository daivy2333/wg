# Tasks: m2-data-exploration-and-preprocessing

> 实现 M2 数据探索与特征工程（任务 2.1-2.10）

## 1. 异常值处理模块（任务 2.4）

- [ ] 1.1 **TDD RED**: 写 `tests/test_outlier.py`，覆盖 detect_outliers_iqr / clip_outliers / log1p_transform / outlier_pipeline
- [ ] 1.2 **Verify RED**: 运行 `pytest tests/test_outlier.py`，展示失败输出（outlier 模块未实现）
- [ ] 1.3 实现 `src/data/outlier.py`：IQR 检测 + clip + log1p + 管线整合
- [ ] 1.4 **Verify GREEN**: 运行 `pytest tests/test_outlier.py`，展示全部通过
- [ ] 1.5 端到端：在真实数据上调用 `outlier_pipeline(load_train())`，展示异常值统计 + 处理后 shape

## 2. 特征选择模块（任务 2.7）

- [ ] 2.1 **TDD RED**: 写 `tests/test_feature_selector.py`，覆盖 variance_threshold / rf_importance / top_k / pipeline
- [ ] 2.2 **Verify RED**: 运行 `pytest tests/test_feature_selector.py`，展示失败输出
- [ ] 2.3 实现 `src/data/feature_selector.py`：VarianceThreshold + RandomForestClassifier(n_estimators=100) + Top-K
- [ ] 2.4 **Verify GREEN**: 运行测试，全部通过
- [ ] 2.5 端到端：在预处理后的 X_train 上调用 pipeline，展示 Top-20 特征名 + 维度对比

## 3. 数据持久化模块（任务 2.9）

- [ ] 3.1 **TDD RED**: 写 `tests/test_persistence.py`，覆盖 save/load_pickle + save_m2_datasets（二分类+多分类）
- [ ] 3.2 **Verify RED**: 运行测试，展示失败输出
- [ ] 3.3 实现 `src/data/persistence.py`：pickle save/load + 整合函数 + 自动创建目录
- [ ] 3.4 **Verify GREEN**: 运行测试，全部通过
- [ ] 3.5 端到端：调用 `save_m2_datasets()` 生成 8 个 .pkl 文件到 `outputs/processed/`，验证 `ls` 输出

## 4. 攻击类型深度分析（任务 2.3）

- [ ] 4.1 在 notebook 中增加 cell：23 种攻击类型频次表（train + test）
- [ ] 4.2 增加 cell：4 大类分布（Normal/DoS/Probe/R2L/U2R）柱状图
- [ ] 4.3 增加 cell：二分类映射统计 + unknown 攻击识别（test 集中训练未见的 14 种）

## 5. 特征分布分析（任务 2.2）

- [ ] 5.1 增加 cell：41 特征 describe 基本统计
- [ ] 5.2 增加 cell：数值特征分布直方图（前 12 数值特征）
- [ ] 5.3 增加 cell：分类特征 top-10 频次柱状图
- [ ] 5.4 增加 cell：相关系数热力图（数值特征子集，避免 53×53 过密）

## 6. 异常值 notebook 分析（任务 2.4 报告层）

- [ ] 6.1 增加 cell：IQR 异常值检测结果（每列超限数量）
- [ ] 6.2 增加 cell：clip 处理前后分布对比
- [ ] 6.3 增加 cell：log1p 变换前后对比（src_bytes/dst_bytes）

## 7. 特征选择 notebook 分析（任务 2.7 报告层）

- [ ] 7.1 增加 cell：方差阈值过滤结果（保留列数对比）
- [ ] 7.2 增加 cell：RF 训练 + feature_importances_ 柱状图
- [ ] 7.3 增加 cell：Top-20 特征列表 + M3/M4 启示

## 8. EDA 报告（任务 2.10）

- [ ] 8.1 创建 `docs/eda_report.md`，5 章节（数据概览/特征分析/异常值/特征选择/结论）
- [ ] 8.2 嵌入关键统计数字（shape、标签分布、Top-20 列表）
- [ ] 8.3 在 M3/M4 结论章节给出训练建议

## 9. Notebook 端到端验证（任务 2.1 完整）

- [ ] 9.1 运行 `jupyter nbconvert --to notebook --execute notebooks/01_data_exploration.ipynb`
- [ ] 9.2 验证 notebook 含 17+ cells 全部 executed

## 10. 验证与归档

- [ ] 10.1 运行 `pytest tests/ -v`，展示全部通过输出（M1 29 + M2 新增）
- [ ] 10.2 验证 `outputs/processed/*.pkl` 8 个文件存在 + 可加载
- [ ] 10.3 运行 `python -c "import src.data.outlier, src.data.feature_selector, src.data.persistence"`，展示无 ImportError
- [ ] 10.4 更新 `.claude/docs/tasks.md` 中 M2 任务 2.1-2.10 状态为 ✅ 已完成
- [ ] 10.5 更新 `.claude/docs/SNAPSHOT.md` 最近修改章节
- [ ] 10.6 运行 `openspec archive m2-data-exploration-and-preprocessing --yes` 归档变更