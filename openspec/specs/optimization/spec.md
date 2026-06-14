# Optimization Spec

> Version: 1.0.0 | Last Updated: 2026-06-14

## Purpose

记录项目中发现的优化点和改进方向，持续提升代码质量和性能。

## Requirements

### Requirement: 优化点记录

发现的性能瓶颈、代码异味、技术债务 SHALL 记录，包含当前影响和建议方案。

#### Scenario: 发现优化机会

- **WHEN** 开发者发现代码中存在性能问题、重复代码、过度复杂设计等
- **THEN** 必须记录到 optimization/spec.md，包含：问题描述、当前影响、建议方案、优先级

#### Scenario: 评估优化价值

- **WHEN** 开发者需要决定是否进行某项优化
- **THEN** 可以参考 optimization/spec.md 中的记录，评估影响范围和收益

### Requirement: 优化完成追踪

已完成的优化 SHALL 记录完成状态，保留历史记录。

#### Scenario: 完成优化

- **WHEN** 开发者完成了某项优化工作
- **THEN** 必须更新 optimization/spec.md，标记为已完成，记录完成日期和实际效果

### Requirement: 优化优先级管理

优化点 SHALL 有优先级排序，合理安排优化顺序。

#### Scenario: 规划优化计划

- **WHEN** 开发者制定优化计划时
- **THEN** 可以参考 optimization/spec.md 中的优先级标注，优先处理高优先级优化点

## 优化记录

| # | 问题描述 | 当前影响 | 建议方案 | 优先级 | 状态 |
|---|----------|----------|----------|--------|------|
| 1 | seaborn/scikit-learn 未安装 | 无法运行 ML 模型训练和高级可视化 | `pip install seaborn scikit-learn` | 高 | 待处理 |
| 2 | 数据集未解压 | 无法开始数据探索 | `unzip dataset/archive.zip -d dataset/` | 高 | 待处理 |
| 3 | 项目未初始化 git | 无版本控制 | `git init` + 初始提交 | 中 | 待处理 |
