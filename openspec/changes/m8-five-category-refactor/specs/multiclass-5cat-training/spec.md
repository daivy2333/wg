# multiclass-5cat-training

> 5 类多分类模型训练
> Version: 1.0.0 | Last Updated: 2026-06-16

## ADDED Requirements

### Requirement: DT/RF 5 类训练
`train_dt_multiclass()` 和 `train_rf_multiclass()` SHALL 在 5 类标签上正常训练，输出 5 类预测。

#### Scenario: DT 5 类训练通过
- **WHEN** 在 5 类标签数据上调用 `train_dt_multiclass(X_train, y_train_5cat)`
- **THEN** 模型 `predict()` 返回 0-4 范围预测值
- **AND** 准确率 > 20%（5 类随机基线）

### Requirement: MLP 5 类训练
`MLPClassifier` SHALL 支持 `output_dim=5`，在 5 类标签上正常训练。

#### Scenario: MLP 5 类训练通过
- **WHEN** 创建 `MLPClassifier(input_dim=20, output_dim=5, hidden_dims=(128,128))`
- **THEN** 模型可正常 fit 和 predict
- **AND** 准确率 > 20%

### Requirement: CNN/LSTM 5 类训练
CNN 和 LSTM 模型 SHALL 支持 5 类输出，利用 GPU 加速训练。

#### Scenario: CNN 5 类训练通过
- **WHEN** 创建 CNN 模型 `output_dim=5`
- **THEN** 模型在 GPU 上正常训练
- **AND** 推理时间 < 1s（22,544 样本）

## MODIFIED Requirements

### Requirement: SMOTE 多分类实验移除
SMOTE 多分类训练 SHALL 从 M8 中移除。原 `mlp_multiclass_smote.pt` 标记为废弃。

#### Scenario: SMOTE 模型从评估中排除
- **WHEN** 运行 `evaluate_m5.py`
- **THEN** 不加载 `mlp_multiclass_smote.pt`
- **AND** 对比表中不出现 SMOTE 行
