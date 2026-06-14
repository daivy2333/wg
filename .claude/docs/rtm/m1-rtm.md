# Requirements Traceability Matrix — m1-infrastructure

> Generated: 2026-06-14
> 用途：确保每个 Requirement 都有 Task 覆盖，Simplification 需用户 approval

## 需求清单（17 条，来自 3 个 specs）

| ID | Requirement | 来源 spec | Scenario 数 |
|----|-------------|-----------|-------------|
| R1 | 标准项目目录结构 | project-skeleton | 2 |
| R2 | 依赖锁定 | project-skeleton | 3 |
| R3 | NSL-KDD 41 特征名常量 | data-loading | 2 |
| R4 | 加载训练集（Happy） | data-loading | 1 |
| R5 | 加载训练集（Sad - FileNotFound） | data-loading | 1 |
| R6 | 加载训练集（Edge - 列数断言） | data-loading | 1 |
| R7 | 加载测试集（Happy） | data-loading | 1 |
| R8 | 加载测试集（Sad - FileNotFound） | data-loading | 1 |
| R9 | 文件编码处理（latin-1） | data-loading | 1 |
| R10 | 数据文件无 header | data-loading | 1 |
| R11 | 缺失值处理 | data-preprocessing | 2 |
| R12 | 分类特征编码（混合策略） | data-preprocessing | 3 |
| R13 | 数值特征标准化（StandardScaler） | data-preprocessing | 3 |
| R14 | 难度列丢弃 | data-preprocessing | 1 |
| R15 | 数据集划分（保留原始 KDD） | data-preprocessing | 1 |
| R16 | 二分类与多分类标签 | data-preprocessing | 2 |
| R17 | 预处理管线整合 | data-preprocessing | 1 |

---

## RTM 矩阵

| Requirement | Task(s) | Coverage | Simplification | Status |
|-------------|---------|----------|----------------|--------|
| R1 标准目录结构 | T1.1, T1.2, T1.3 | 100% | 无 | ✅ |
| R2 依赖锁定 | T2.1, T2.2 | 100% | 无 | ✅ |
| R3 41 特征名常量 | T3.3 | 100% | 无 | ✅ |
| R4 加载训练集 Happy | T3.3, T3.4, T3.5 | 100% | 无 | ✅ |
| R5 加载训练集 Sad | T3.1, T3.3, T3.4 | 100% | 无 | ✅ |
| R6 加载训练集 Edge | T3.1, T3.3, T3.4 | 100% | 无 | ✅ |
| R7 加载测试集 Happy | T3.3, T3.4, T3.5 | 100% | 无 | ✅ |
| R8 加载测试集 Sad | T3.1, T3.3, T3.4 | 100% | 无 | ✅ |
| R9 latin-1 编码 | T3.3, T3.4 | 100% | 无 | ✅ |
| R10 无 header | T3.3, T3.4 | 100% | 无 | ✅ |
| R11 缺失值处理 | T4.1, T4.3, T4.4 | 100% | "缺失值非零不自动填充" ⚠️ | ⚠️ |
| R12 分类特征编码 | T4.1, T4.3, T4.4 | 100% | 无 | ✅ |
| R13 数值特征标准化 | T4.1, T4.3, T4.4 | 100% | 无 | ✅ |
| R14 难度列丢弃 | T4.3, T4.4 | 100% | 无 | ✅ |
| R15 数据集划分 | T4.3, T4.4 | 100% | 无 | ✅ |
| R16 二分类/多分类标签 | T4.1, T4.3, T4.4 | 100% | 无 | ✅ |
| R17 预处理管线整合 | T4.1, T4.3, T4.4 | 100% | 无 | ✅ |

---

## ⚠️ Simplification 待审批

### S1: 缺失值非零不自动填充

- **原方案（spec）**: `check_missing()` 检测到缺失值后**不自动填充**，输出警告日志，由调用方决定策略
- **风险**: 若实际数据有缺失值（虽然 NSL-KDD 官方称无），M1 阶段不处理会导致 M2 EDA 报错
- **简化依据**: NSL-KDD 官方数据无缺失值是已知前提；M2 阶段如发现可重新加入处理逻辑
- **替代方案**: 改为自动填充（fillna(0)），但引入静默修复风险
- **需要**: 用户 approval

---

## 状态汇总

| 状态 | 数量 | 占比 |
|------|------|------|
| ✅ Covered | 16 | 94.1% |
| ⚠️ Simplified | 1 | 5.9% |
| ❌ Missing | 0 | 0% |

**Gate 2 通过条件**：所有 ✅ + 所有 ⚠️ 已获 user approval + 无 ❌

---

## 下游影响

| Requirement | 下游影响（M2-M5） |
|-------------|------------------|
| R1-R2 | M2 启动必要条件 |
| R3-R10 | M2 EDA 直接使用 loader |
| R11-R17 | M3/M4 训练直接使用 preprocess_pipeline |
| R5/R8 | 用户路径错误友好提示（不影响功能） |