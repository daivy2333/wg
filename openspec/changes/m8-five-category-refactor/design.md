## Context

M7 终稿使用 40 类细粒度攻击标签做多分类，准确率 4.6-10%。分析确认低准确率源自 NSL-KDD 数据集设计（15 类未见攻击 + 极端类不平衡），非代码缺陷。已验证后处理映射为 5 类无效（+2pp）。决策：在训练阶段切换为 5 大类标签，重训所有多分类模型。

GPU: RTX 4060 Laptop (8GB VRAM, CUDA 可用)，CNN/LSTM 训练可行。

## Goals / Non-Goals

**Goals:**
- `make_labels()` 新增 `multiclass_5cat` 模式，利用 `label_id_to_category.json` 生成 5 类标签
- 重训 DT/RF/MLP/CNN/LSTM 的 5 类多分类模型
- 重跑 M5 评估，产出 5 类对比指标和图表
- 论文全章更新为 5 类叙事

**Non-Goals:**
- 不修改二分类实验
- 不改变 CJC 排版
- 不新增参考文献

## Decisions

### D1: 5 类标签生成方式

**选择**: `make_labels(task="multiclass_5cat")` → 先用原 40 类 LabelEncoder 编码标签名，再查 `label_id_to_category.json` 映射为 0-4

**原因**: 复用已有映射表，不引入新依赖；保持与 40 类编码路径一致

**替代方案**: 直接按字符串标签名映射 → 拒绝，需维护两套编码逻辑

### D2: SMOTE 处理

**选择**: 移除 SMOTE 多分类实验

**原因**: 
1. 5 分类下类别不平衡程度大幅缓解（最小类 U2R 仍有 52 训练样本 vs 5 类总数远少于 40 类）
2. 原 SMOTE 已证明负面效果，5 分类下预期同样无效
3. 论文中 SMOTE 讨论在 appendix 保留为历史记录

### D3: 模型重训顺序

**选择**: 数据→DT→RF→MLP→CNN→LSTM（串行），评估最后

**原因**: DT/RF 是 sklearn 模型（CPU），可并行；MLP/CNN/LSTM 共享 GPU 需串行避免 OOM。实际执行：DT+RF 并行，MLP→CNN→LSTM 串行

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| 5 类准确率仍不理想 | 5 类随机基线 20%，已知类准确率预期 70%+；若低于 50% 需排查标签映射 |
| 重训时间过长 | DT/RF 分钟级；MLP ~10min；CNN/LSTM ~20min (GPU) |
| 现有 pickle 文件被覆盖 | 备份原 `y_train_multi.pkl` 为 `y_train_multi_40class.pkl` |