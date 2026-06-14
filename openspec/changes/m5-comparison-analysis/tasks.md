## 1. 评估基础设施（src/evaluation/）

- [ ] 1.1 创建 `src/evaluation/metrics.py`——实现 `compute_binary_metrics(y_true, y_pred, y_prob) -> dict`，返回 accuracy/precision/recall/f1/auc
- [ ] 1.2 实现 `compute_multiclass_metrics(y_true, y_pred, num_classes, unseen_ids) -> dict`，返回 full_accuracy/known_class_accuracy/f1_macro
- [ ] 1.3 实现 `compute_f1_by_category(y_true, y_pred, label_to_category_map) -> dict[str, float]`，返回各攻击大类的 F1
- [ ] 1.4 创建 `src/evaluation/__init__.py`——导出以上公开 API
- [ ] 1.5 QA：运行 `python -c "from src.evaluation import compute_binary_metrics; print(compute_binary_metrics([0,0,1,1],[0,1,0,1],[0.1,0.4,0.6,0.9]))"` 验证 basic 指标计算正确

## 2. 可视化函数（src/evaluation/plot.py）

- [ ] 2.1 创建 `src/evaluation/plot.py`——实现 `plot_confusion_matrix_heatmap(y_true, y_pred, title, save_path)`，生成 2×2 热力图
- [ ] 2.2 实现 `plot_roc_curves(model_roc_dict, save_path)`——多模型 ROC 曲线叠加图
- [ ] 2.3 实现 `plot_f1_by_category_bars(f1_dicts, save_path)`——攻击大类 F1 分组柱状图
- [ ] 2.4 实现 `plot_feature_importance_comparison(dt_imp, rf_imp, top_k, save_path)`——特征重要度并排对比
- [ ] 2.5 实现 `plot_dl_vs_ml_comparison(metrics_dict, save_path)`——深度学习 vs 传统 ML 对比图
- [ ] 2.6 QA：验证 `outputs/figures/` 目录下文件存在且非空（在编排脚本中验证）

## 3. 标签映射生成

- [ ] 3.1 在 `scripts/evaluate_m5.py` 中实现 `derive_label_mapping()`——从原始 CSV 重新拟合 LabelEncoder，生成 `outputs/label_id_to_name.json`
- [ ] 3.2 生成 `outputs/label_id_to_category.json`——将 attack name 映射到 DoS/Probe/R2L/U2R/Normal 五类（复用 M2 的 ATTACK_CATEGORY dict）
- [ ] 3.3 QA：`python -c "import json; m=json.load(open('outputs/label_id_to_name.json')); assert len(m) >= 23"`

## 4. 模型加载与评估编排

- [ ] 4.1 在 `scripts/evaluate_m5.py` 中定义 `MODEL_CONFIGS` dict——精确记录 6 个 PyTorch 模型的 class 和 kwargs
- [ ] 4.2 实现 `load_all_models()`——加载全部 10 个模型（4 sklearn + 6 PyTorch），返回 dict
- [ ] 4.3 实现 `evaluate_all_binary(models, X_test, y_test) -> dict`——对所有二分类模型运行 predict + 指标计算
- [ ] 4.4 实现 `evaluate_all_multiclass(models, X_test_multi, y_test_multi, unseen_ids) -> dict`——对所有多分类模型运行 predict + 双精度计算
- [ ] 4.5 保存 `outputs/metrics_m5.json`——包含所有模型的完整指标
- [ ] 4.6 QA：`python -c "import json; d=json.load(open('outputs/metrics_m5.json')); assert 'dt_binary' in d; assert abs(d['rf_binary']['f1'] - 0.688) < 0.1"` 验证指标合理性

## 5. 图表生成（任务 5.2–5.6）

- [ ] 5.1 任务 5.2：生成 DT/RF/MLP 二分类混淆矩阵热力图（3 张，`outputs/figures/09_*.png`）
- [ ] 5.2 任务 5.3：生成攻击大类 F1 分组柱状图（1 张，`outputs/figures/10_f1_by_category.png`）
- [ ] 5.3 任务 5.4：生成二分类 ROC 曲线叠加图（1 张，`outputs/figures/11_roc_curves.png`）
- [ ] 5.4 任务 5.5：生成 DT vs RF 特征重要度并排对比图（1 张，`outputs/figures/12_feature_importance_comparison.png`）
- [ ] 5.5 任务 5.6：生成深度学习 vs 传统 ML 关键指标对比图（1 张，`outputs/figures/13_dl_vs_ml_comparison.png`）
- [ ] 5.6 QA：`for f in outputs/figures/09_*png outputs/figures/1[0-3]_*png; do test -s "$f" && echo "OK: $f" || echo "MISSING: $f"; done`

## 6. 对比分析报告（任务 5.7）

- [ ] 6.1 编写报告 §1-2（概述 + 指标汇总表）——含所有模型二分类+多分类指标 Markdown 表格
- [ ] 6.2 编写报告 §3-5（混淆矩阵分析 + 攻击大类对比 + ROC 分析）——引用对应图表
- [ ] 6.3 编写报告 §6-7（特征重要度对比 + 深度学习 vs 传统 ML）——引用对应图表
- [ ] 6.4 编写报告 §8（SMOTE 失败实验分析）——引用 O-NN-01，诚实记录副作用
- [ ] 6.5 编写报告 §9（总结与局限）——含关键结论、unseen attack 局限、未来工作
- [ ] 6.6 QA：验证 `docs/comparison_report.md` 存在且包含 9 个章节标题（`grep "^## " docs/comparison_report.md | wc -l` >= 9）

## 7. 编排脚本集成与端到端测试

- [ ] 7.1 实现 `scripts/evaluate_m5.py` 的 `main()` 函数——串联全部流程（加载→评估→图表→报告）
- [ ] 7.2 添加 WSL 兼容代码（OMP=1, MKL=1, CUDA="" , torch.set_num_threads(1)）
- [ ] 7.3 端到端 QA：运行 `python scripts/evaluate_m5.py`，验证 exit code = 0，所有输出文件存在
