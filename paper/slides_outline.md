# 答辩PPT提纲 — NSL-KDD网络入侵检测系统

> 总计 15 张幻灯片，覆盖 6 章内容，章节比例：
> 第 1 章 3 张 / 第 2 章 2 张 / 第 3 章 2 张 / 第 4 章 3 张 / 第 5 章 3 张 / 第 6 章 2 张
> 每张幻灯片列出：内容要点、配套图、讲解时长（合计约 18 分钟）

## Slide 1: 标题页
- 论文题目：基于NSL-KDD数据集的网络入侵检测研究
- 副标题：传统机器学习与深度学习方法对比
- 三人姓名：A 同学（数据）/ B 同学（传统 ML）/ C 同学（深度学习 + 对比 + 论文）
- 日期：2026-06
- 配套图：无
- 时长：30 秒

## Slide 2: 研究背景与意义
- 网络入侵检测的重要性
- 传统签名检测方法的局限
- 机器学习方法的优势

## Slide 3: NSL-KDD数据集概览
- 训练集125,973条 / 测试集22,544条
- 41个特征，4大类攻击（DoS/Probe/R2L/U2R）+ Normal
- 关键挑战：测试集含15种训练未见攻击
- 图：02_attack_category_distribution.png

## Slide 4: 数据预处理流程
- 41维 → 53维(OneHot) → 32维(方差过滤) → 20维(RF重要度)
- log1p变换处理长尾特征
- IQR+clip异常值策略
- 图：07_log1p_before_after.png, 08_feature_importance_top20.png

## Slide 5: 决策树模型
- 网格搜索：30组合（max_depth × min_samples_split × criterion）
- 最优参数：entropy / max_depth=20 / min_samples_split=2
- 测试集：acc=0.776, f1=0.762, AUC=0.782
- 推理速度最快（0.02s）
- 图：09_confusion_matrix_dt.png

## Slide 6: 随机森林模型
- 网格搜索：24组合（n_estimators × max_depth × min_samples_split）
- 最优参数：300棵树 / max_depth=20
- 测试集：acc=0.723, f1=0.688, AUC=0.915
- 概率校准最优
- 图：09_confusion_matrix_rf.png

## Slide 7: MLP神经网络
- 架构：20→128→64→2，ReLU+Dropout 0.3
- 验证集：f1=0.989, AUC=0.999
- 测试集：f1=0.720（验证/测试鸿沟26%）
- 调优边际收益＜0.2%
- 图：09_confusion_matrix_mlp.png

## Slide 8: CNN与LSTM探索
- CNN 1D：acc=0.764, f1=0.745, AUC=0.882
- LSTM：acc=0.743, f1=0.716, AUC=0.883
- 结论：表格数据缺乏空间/时序结构，CNN/LSTM未超越MLP

## Slide 9: SMOTE实验（负结果）
- U2R仅52条，R2L仅995条
- SMOTE过采样后多分类准确率从0.101降至0.028（-72%）
- 结论：极端小样本下合成数据引入伪模式，不推荐使用

## Slide 10: 全模型二分类对比
- 6个模型在统一测试集上的完整对比
- 图：13_dl_vs_ml_comparison.png

## Slide 11: ROC曲线与攻击类别分析
- 5条ROC曲线叠加对比
- 四大攻击类别F1拆解（DoS→U2R递减）
- 图：11_roc_curves.png, 10_f1_by_category.png

## Slide 12: 特征重要度对比
- DT与RF在Top-15特征上高度一致
- flag_SF同为最重要特征（14.8%）
- 图：12_feature_importance_comparison.png

## Slide 13: 核心结论
- DT在测试集上综合最优（acc=0.776, f1=0.762）
- 神经网络在IID验证集上突出，但在OOD测试集上优势消失
- 多分类瓶颈：15种unseen攻击
- SMOTE在本数据集上失败
- “模型复杂度不等于性能提升”

## Slide 14: 研究局限与未来工作
- 局限：单数据集、无统计检验、CNN/LSTM架构不匹配
- 未来：开放集识别、Focal Loss、集成学习、跨数据集验证、实时部署

## Slide 15: 致谢与 Q&A
- 致谢指导老师与同学
- Q&A 开放提问
- 配套图：无
- 时长：弹性

---

## 配套图清单

| 图编号 | 文件 | 使用幻灯片 |
|---|---|---|
| 02 | `02_attack_category_distribution.png` | Slide 3 |
| 07 | `07_log1p_before_after.png` | Slide 4 |
| 08 | `08_feature_importance_top20.png` | Slide 4 |
| 09-dt | `09_confusion_matrix_dt.png` | Slide 5 |
| 09-rf | `09_confusion_matrix_rf.png` | Slide 6 |
| 09-mlp | `09_confusion_matrix_mlp.png` | Slide 7 |
| 10 | `10_f1_by_category.png` | Slide 11 |
| 11 | `11_roc_curves.png` | Slide 11 |
| 12 | `12_feature_importance_comparison.png` | Slide 12 |
| 13 | `13_dl_vs_ml_comparison.png` | Slide 10 |

## 章节覆盖检查

| 章节 | 幻灯片 | 占比 |
|---|---|---|
| 第 1 章 绪论 | Slide 2, 3 | 2/15（13%） |
| 第 2 章 预处理 | Slide 4 | 1/15（7%） |
| 第 3 章 传统 ML | Slide 5, 6 | 2/15（13%） |
| 第 4 章 深度学习 | Slide 7, 8, 9 | 3/15（20%） |
| 第 5 章 对比分析 | Slide 10, 11, 12 | 3/15（20%） |
| 第 6 章 总结展望 | Slide 13, 14, 15 | 3/15（20%） |
| 标题与背景 | Slide 1 | 1/15（7%） |

> 说明：第 2 章预处理重点在工程实现，PPT 上以 1 张流水线图 + 1 张特征重要度图覆盖即可；研究重点（深度学习与对比分析）各占 3 张以保证讨论深度。
