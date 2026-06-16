# multiclass-5cat-paper

> 论文 5 类多分类章节更新
> Version: 1.0.0 | Last Updated: 2026-06-16

## ADDED Requirements

### Requirement: 5 类多分类叙事
论文章节 SHALL 将多分类描述从"40 种攻击标签"改为"5 大类攻击类别（Normal/DoS/Probe/R2L/U2R）"。

#### Scenario: ch1 描述一致
- **WHEN** 读取 ch1 多分类目标描述
- **THEN** 表述为"5 大攻击类别的多分类识别能力"
- **AND** 不出现"40 类"或"40 种"字样

#### Scenario: ch3/ch4 指标一致
- **WHEN** 读取 ch3/ch4 多分类实验数据
- **THEN** 准确率/随机基线等数字与重训结果一致
- **AND** 随机基线为 1/5=20%

### Requirement: 删除 unseen 攻击讨论
5 分类无 unseen 攻击问题，SHALL 从多分类章节中移除相关讨论。

#### Scenario: ch3/ch4 无 unseen 表述
- **WHEN** 搜索 ch3/ch4 多分类段落
- **THEN** 无"unseen 攻击""训练未见""零样本"等表述

### Requirement: 更新 per-class F1 分析
ch5 SHALL 使用 5 类 per-class F1 替代原 40 类分析，图表使用 5×5 混淆矩阵。

#### Scenario: NxN混淆矩阵
- **WHEN** 查看图 9/11/12
- **THEN** 混淆矩阵为 5×5（原为 N×N，N>5）
