"""M3 训练编排脚本：DT + RF 基线 + 网格搜索 + 持久化 + 报告生成

运行：
    /home/daivy/miniconda3/bin/python scripts/train_m3.py

输入：
    outputs/processed/{X,y}_{train,test}.pkl（二分类 + 多分类双套）

输出：
    outputs/models/{dt,rf}_{binary,multiclass}_best.joblib
    docs/model_report_dt_rf.md

性能 vs 稳定性权衡：
    WSL 报告 32 逻辑核但物理资源可能只分配 2-4 核，n_jobs=-1
    会 spawn 过多进程导致 OOM/fork 失败（踩坑-003）。

    本脚本使用动态公式 n_jobs = min(4, max(1, cpu_count // 8))：
      - 32 核机器 → 4 worker（并行加速）
      - 4-8 核机器 → 1 worker（退化为单进程但稳）
    既保证并行收益，又避免资源爆炸。

    tests/conftest.py 仍强制 OMP=1 + 测试函数默认 n_jobs=1。
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 动态计算 n_jobs：32 核 → 4，8 核 → 1，硬上限 4
_CPU_COUNT = os.cpu_count() or 1
N_JOBS = min(4, max(1, _CPU_COUNT // 8))
print(f"检测到 CPU 核数: {_CPU_COUNT}, 本脚本 n_jobs = {N_JOBS}")

from src.data.persistence import load_pickle
from src.models.decision_tree import (
    train_dt_binary,
    train_dt_multiclass,
    grid_search_dt,
    evaluate_model,
)
from src.models.random_forest import (
    train_rf_binary,
    train_rf_multiclass,
    grid_search_rf,
    get_top_k_features,
)
from src.models.persistence import save_best_models

PROCESSED_DIR = ROOT / "outputs" / "processed"
MODELS_DIR = ROOT / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


# ============================================================
# 1. 加载 M2 pickle 数据
# ============================================================
banner("Step 1: 加载 M2 pickle 数据")
X_train_bin = load_pickle(PROCESSED_DIR / "X_train.pkl")
X_test_bin = load_pickle(PROCESSED_DIR / "X_test.pkl")
y_train_bin = load_pickle(PROCESSED_DIR / "y_train.pkl")
y_test_bin = load_pickle(PROCESSED_DIR / "y_test.pkl")
X_train_multi = load_pickle(PROCESSED_DIR / "X_train_multi.pkl")
X_test_multi = load_pickle(PROCESSED_DIR / "X_test_multi.pkl")
y_train_multi = load_pickle(PROCESSED_DIR / "y_train_multi.pkl")
y_test_multi = load_pickle(PROCESSED_DIR / "y_test_multi.pkl")
print(f"二分类: train={X_train_bin.shape}, test={X_test_bin.shape}")
print(f"多分类: train={X_train_multi.shape}, test={X_test_multi.shape}")


# ============================================================
# 2. DT 二分类基线
# ============================================================
banner("Step 2: DT 二分类基线训练")
dt_bin = train_dt_binary(X_train_bin, y_train_bin)
m_dt_bin = evaluate_model(dt_bin, X_test_bin, y_test_bin, task="binary")
print(f"DT 二分类基线: {m_dt_bin}")


# ============================================================
# 3. DT 二分类网格搜索
# ============================================================
banner("Step 3: DT 二分类网格搜索（30 组合 × 5 折）")
start = time.time()
best_dt_bin, params_dt_bin, score_dt_bin = grid_search_dt(
    X_train_bin, y_train_bin, task="binary", cv=5
)
print(f"耗时 {time.time()-start:.1f}s, best_params={params_dt_bin}, best_cv_f1={score_dt_bin:.4f}")
m_best_dt_bin = evaluate_model(best_dt_bin, X_test_bin, y_test_bin, task="binary")
print(f"DT 二分类 best test metrics: {m_best_dt_bin}")


# ============================================================
# 4. RF 二分类基线
# ============================================================
banner("Step 4: RF 二分类基线训练（n_jobs=-1）")
rf_bin = train_rf_binary(X_train_bin, y_train_bin, n_jobs=N_JOBS)
m_rf_bin = evaluate_model(rf_bin, X_test_bin, y_test_bin, task="binary")
print(f"RF 二分类基线: {m_rf_bin}")


# ============================================================
# 5. RF 二分类网格搜索
# ============================================================
banner("Step 5: RF 二分类网格搜索（24 组合 × 5 折，n_jobs=-1）")
start = time.time()
best_rf_bin, params_rf_bin, score_rf_bin = grid_search_rf(
    X_train_bin, y_train_bin, task="binary", cv=5, n_jobs=N_JOBS
)
print(f"耗时 {time.time()-start:.1f}s, best_params={params_rf_bin}, best_cv_f1={score_rf_bin:.4f}")
m_best_rf_bin = evaluate_model(best_rf_bin, X_test_bin, y_test_bin, task="binary")
print(f"RF 二分类 best test metrics: {m_best_rf_bin}")


# ============================================================
# 6. RF 特征重要度
# ============================================================
banner("Step 6: RF 特征重要度 Top-20")
top20 = get_top_k_features(best_rf_bin, k=20)
print("Top-20 特征（RF 网格搜索后）:")
for i, (feat, imp) in enumerate(top20.items(), 1):
    print(f"  {i:2d}. {feat}: {imp:.4f}")


# ============================================================
# 7. DT/RF 多分类
# ============================================================
banner("Step 7: DT 多分类训练")
dt_multi = train_dt_multiclass(X_train_multi, y_train_multi)
m_dt_multi = evaluate_model(dt_multi, X_test_multi, y_test_multi, task="multiclass")
print(f"DT 多分类: {m_dt_multi}")

banner("Step 7b: RF 多分类训练（n_jobs=-1）")
rf_multi = train_rf_multiclass(X_train_multi, y_train_multi, n_jobs=N_JOBS)
m_rf_multi = evaluate_model(rf_multi, X_test_multi, y_test_multi, task="multiclass")
print(f"RF 多分类: {m_rf_multi}")


# ============================================================
# 8. 模型持久化
# ============================================================
banner("Step 8: 模型持久化")
save_best_models(
    dt_binary=best_dt_bin,
    rf_binary=best_rf_bin,
    dt_multiclass=dt_multi,
    rf_multiclass=rf_multi,
    output_dir=MODELS_DIR,
)
print(f"✓ 4 个模型保存到 {MODELS_DIR}/")
for p in sorted(MODELS_DIR.glob("*.joblib")):
    size_kb = p.stat().st_size / 1024
    print(f"  {p.name}: {size_kb:.1f} KB")


print("\n✓ M3 训练完成")