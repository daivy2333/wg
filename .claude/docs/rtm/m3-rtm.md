# Requirements Traceability Matrix — m3-traditional-ml

> Generated: 2026-06-14
> 用途：M3 需求追踪矩阵

## 需求清单（13 条，来自 4 个 specs）

| ID | Requirement | 来源 spec | Scenario 数 |
|----|-------------|-----------|-------------|
| R1 | DT 二分类基线训练 | decision-tree-training | 2 |
| R2 | DT 多分类训练 | decision-tree-training | 2 |
| R3 | DT GridSearchCV 调优 | decision-tree-training | 2 |
| R4 | class_weight='balanced' | decision-tree-training | 1 |
| R5 | RF 二分类基线训练 | random-forest-training | 2 |
| R6 | RF 多分类训练 | random-forest-training | 1 |
| R7 | RF GridSearchCV 调优 | random-forest-training | 2 |
| R8 | RF 特征重要度 | random-forest-training | 2 |
| R9 | 4 大攻击类别 F1 对比 | random-forest-training | 1 |
| R10 | joblib 模型持久化 | model-persistence | 3 |
| R11 | M3 最佳模型持久化 | model-persistence | 2 |
| R12 | M3 模型训练报告 | ml-model-reporting | 7 |
| R13 | M3 训练脚本 | ml-model-reporting | 1 |

---

## RTM 矩阵

| Requirement | Task(s) | Coverage | Simplification | Status |
|-------------|---------|----------|----------------|--------|
| R1 DT 二分类基线 | T1.1, T1.3, T1.4, T1.5, T4.1, T4.2 | 100% | 无 | ✅ |
| R2 DT 多分类 | T1.1, T1.3, T1.4, T1.5, T4.1, T4.2 | 100% | 无 | ✅ |
| R3 DT 网格搜索 | T1.1, T1.3, T1.4, T1.5, T4.1, T4.2 | 100% | 无 | ✅ |
| R4 class_weight='balanced' | T1.3, T1.4, T2.3, T2.4 | 100% | 无 | ✅ |
| R5 RF 二分类基线 | T2.1, T2.3, T2.4, T2.5, T4.1, T4.2 | 100% | 无 | ✅ |
| R6 RF 多分类 | T2.1, T2.3, T2.4, T2.5, T4.1, T4.2 | 100% | 无 | ✅ |
| R7 RF 网格搜索 | T2.1, T2.3, T2.4, T2.5, T4.1, T4.2 | 100% | 无 | ✅ |
| R8 RF 特征重要度 | T2.1, T2.3, T2.4, T2.5, T4.1, T4.2 | 100% | 无 | ✅ |
| R9 4 大攻击类别 F1 | T2.1, T2.3, T2.4, T5.1, T5.2 | 100% | 无 | ✅ |
| R10 joblib 持久化 | T3.1, T3.3, T3.4, T3.5 | 100% | 无 | ✅ |
| R11 4 模型保存 | T3.1, T3.3, T3.4, T3.5, T4.1, T4.2 | 100% | 无 | ✅ |
| R12 M3 模型报告 | T5.1, T5.2, T5.3 | 100% | 无 | ✅ |
| R13 M3 训练脚本 | T4.1, T4.2 | 100% | 无 | ✅ |

---

## 状态汇总

| 状态 | 数量 | 占比 |
|------|------|------|
| ✅ Covered | 13 | 100% |
| ⚠️ Simplified | 0 | 0% |
| ❌ Missing | 0 | 0% |

**Gate 2 通过条件**：所有 ✅ + 无 ⚠️ 需审批 + 无 ❌ → ✅ 直接通过

---

## 下游影响

| Requirement | 下游影响（M4/M5） |
|-------------|------------------|
| R1-R4 DT 训练 | M5 对比基线 + M4 借鉴网格搜索流程 |
| R5-R9 RF 训练 | M4（MLP）与传统 ML 对比基准 |
| R10-R11 持久化 | M4 直接 joblib.load 最佳模型 |
| R12-R13 报告 | M6 论文直接引用 |