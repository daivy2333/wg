# Tasks: m3-traditional-ml

> 实现 M3 传统 ML 模型训练（任务 3.1-3.9）

## 1. 决策树训练模块（任务 3.1/3.3）

- [ ] 1.1 **TDD RED**: 写 `tests/test_decision_tree.py`，覆盖 train_dt_binary / train_dt_multiclass / grid_search_dt / class_weight
- [ ] 1.2 **Verify RED**: 展示失败输出
- [ ] 1.3 实现 `src/models/decision_tree.py`：基线 + 网格搜索 + 多分类
- [ ] 1.4 **Verify GREEN**: 全部测试通过
- [ ] 1.5 端到端：在 X_train.pkl 上调用 grid_search_dt，展示最优参数

## 2. 随机森林训练模块（任务 3.4/3.6/3.7）

- [ ] 2.1 **TDD RED**: 写 `tests/test_random_forest.py`，覆盖 train_rf_binary / train_rf_multiclass / grid_search_rf / get_top_k_features / f1_by_category
- [ ] 2.2 **Verify RED**: 展示失败输出
- [ ] 2.3 实现 `src/models/random_forest.py`：基线 + 网格搜索 + 重要度 + F1 拆解
- [ ] 2.4 **Verify GREEN**: 全部测试通过
- [ ] 2.5 端到端：在 X_train.pkl 上调用 grid_search_rf，展示最优参数 + Top-20

## 3. 模型持久化（任务 3.8）

- [ ] 3.1 **TDD RED**: 写 `tests/test_model_persistence.py`，覆盖 save/load_model + save_best_models
- [ ] 3.2 **Verify RED**: 展示失败输出
- [ ] 3.3 实现 `src/models/persistence.py`：joblib save/load + 4 模型整合
- [ ] 3.4 **Verify GREEN**: 全部测试通过
- [ ] 3.5 端到端：调用 save_best_models 生成 4 个 .joblib 到 outputs/models/

## 4. M3 训练编排脚本（任务 3.1-3.8 全流程）

- [ ] 4.1 实现 `scripts/train_m3.py`：加载 pickle → DT 基线 → DT 网格 → RF 基线 → RF 网格 → 持久化 → 报告生成
- [ ] 4.2 运行 `python scripts/train_m3.py`，展示 4 模型训练 + 持久化成功

## 5. M3 模型报告（任务 3.9）

- [ ] 5.1 创建 `docs/model_report_dt_rf.md`，8 章节（数据 / DT 基线 / DT 调优 / RF 基线 / RF 调优 / 4 大类 F1 / 结论）
- [ ] 5.2 嵌入关键数字（最优参数 + 5 指标 + 4 大类 F1 对比表）
- [ ] 5.3 在 M4/M5 启示章节给出建议

## 6. 验证与归档

- [ ] 6.1 运行 `pytest tests/ -v`，展示全部通过（M1 29 + M2 28 + M3 新增）
- [ ] 6.2 验证 `outputs/models/*.joblib` 4 个文件存在 + 可加载
- [ ] 6.3 更新 `.claude/docs/tasks.md` 中 M3 任务 3.1-3.9 状态为 ✅
- [ ] 6.4 更新 `.claude/docs/SNAPSHOT.md` 最近修改章节
- [ ] 6.5 运行 `openspec archive m3-traditional-ml --yes` 归档