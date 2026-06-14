# Requirements Traceability Matrix — m2-data-exploration-and-preprocessing

> Generated: 2026-06-14
> 用途：M2 需求追踪矩阵，11 需求 100% 覆盖

## 需求清单（11 条，来自 4 个 specs）

| ID | Requirement | 来源 spec | Scenario 数 |
|----|-------------|-----------|-------------|
| R1 | IQR 异常值检测 | outlier-handling | 2 |
| R2 | IQR + clip 异常值替换 | outlier-handling | 2 |
| R3 | log1p 长尾变换 | outlier-handling | 2 |
| R4 | 异常值处理管线整合 | outlier-handling | 2 |
| R5 | 方差阈值过滤 | feature-selection | 2 |
| R6 | 随机森林特征重要度 | feature-selection | 2 |
| R7 | 双保险特征选择管线 | feature-selection | 3 |
| R8 | pickle 持久化 | data-persistence | 4 |
| R9 | M2 数据集持久化（双套）| data-persistence | 3 |
| R10 | EDA 报告生成 | eda-reporting | 6 |
| R11 | EDA 脚本可从头运行 | eda-reporting | 2 |

---

## RTM 矩阵

| Requirement | Task(s) | Coverage | Simplification | Status |
|-------------|---------|----------|----------------|--------|
| R1 IQR 异常值检测 | T1.1, T1.3, T1.4, T1.5 | 100% | 无 | ✅ |
| R2 IQR + clip 替换 | T1.1, T1.3, T1.4, T6.2 | 100% | 无 | ✅ |
| R3 log1p 长尾变换 | T1.1, T1.3, T1.4, T6.3 | 100% | 无 | ✅ |
| R4 异常值处理管线整合 | T1.1, T1.3, T1.4, T1.5 | 100% | 无 | ✅ |
| R5 方差阈值过滤 | T2.1, T2.3, T2.4, T7.1 | 100% | 无 | ✅ |
| R6 随机森林特征重要度 | T2.1, T2.3, T2.4, T7.2 | 100% | 无 | ✅ |
| R7 双保险特征选择管线 | T2.1, T2.3, T2.4, T2.5, T7.3 | 100% | 无 | ✅ |
| R8 pickle 持久化 | T3.1, T3.3, T3.4, T3.5 | 100% | 无 | ✅ |
| R9 M2 数据集持久化双套 | T3.1, T3.3, T3.4, T3.5, T10.2 | 100% | 无 | ✅ |
| R10 EDA 报告生成 | T8.1, T8.2, T8.3, T10.5 | 100% | 无 | ✅ |
| R11 EDA 脚本可从头运行 | T4.1-4.3, T5.1-5.4, T6.1-6.3, T7.1-7.3, T9.1, T9.2 | 100% | 无 | ✅ |

---

## 状态汇总

| 状态 | 数量 | 占比 |
|------|------|------|
| ✅ Covered | 11 | 100% |
| ⚠️ Simplified | 0 | 0% |
| ❌ Missing | 0 | 0% |

**Gate 2 通过条件**：所有 ✅ + 无 ⚠️ 需审批 + 无 ❌ → ✅ 直接通过

---

## 下游影响

| Requirement | 下游影响（M3/M4/M5） |
|-------------|---------------------|
| R1-R4 异常值 | M3 决策树天然鲁棒但 M4 MLP 显著受益 |
| R5-R7 特征选择 | M3/M4 训练速度 ↑，Top-20 可解释性强 |
| R8-R9 数据持久化 | M3/M4 直接 pickle.load，零重复计算 |
| R10-R11 EDA | 论文 M6 引用 + M5 对比基础 |