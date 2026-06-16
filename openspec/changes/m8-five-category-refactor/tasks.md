# Tasks: M8 五分类重构

> 将多分类从 40 类攻击标签改为 5 大类，重训模型，更新论文
> GPU: RTX 4060 Laptop (CUDA)

---

## 1. 代码修改

- [ ] 1.1 `make_labels()` 新增 `task="multiclass_5cat"` 分支，利用 `label_id_to_category.json` 生成 5 类标签
- [ ] 1.2 `preprocess_pipeline()` 支持 `task="multiclass_5cat"`
- [ ] 1.3 模型训练脚本 `output_dim` 改为 5（DT/RF/MLP/CNN/LSTM）
- [ ] 1.4 `evaluate_m5.py` 适配 5 类（移除 `unseen_ids`，混淆矩阵 5×5）
- [ ] 1.5 更新测试断言中的 `output_dim` 期望值

---

## 2. 重训模型

- [ ] 2.1 备份原 40 类 pickle 文件，重新运行预处理生成 5 类标签
- [ ] 2.2 重训 DT/RF 多分类（`scripts/train_m3.py`）
- [ ] 2.3 重训 MLP 多分类（基线+调优），移除 SMOTE 版本
- [ ] 2.4 CNN/LSTM 5 分类训练（GPU）
- [ ] 2.5 重跑 `evaluate_m5.py`，验证新指标 > 20%

---

## 3. 论文更新

- [ ] 3.1 ch1：多分类目标改为 5 大类
- [ ] 3.2 ch3：DT/RF 多分类改为 5 类，更新准确率
- [ ] 3.3 ch4：MLP/CNN/LSTM 多分类改为 5 类，删除 SMOTE
- [ ] 3.4 ch5：5 类对比分析 + per-class F1
- [ ] 3.5 ch6：更新局限性（删除 unseen 攻击讨论 + 后处理声明）
- [ ] 3.6 appendix：更新超参数表，SMOTE 移为历史附录

---

## 4. 测试验证

- [ ] 4.1 `pytest tests/ -v` 全部通过
- [ ] 4.2 5 类准确率验证：DT/RF > 60%，MLP > 70%
- [ ] 4.3 论文编译验证：`latexmk -xelatex cjc-main.tex`
