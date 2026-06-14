"""单独运行 SMOTE 步骤并更新 metrics_m4.json（避免重跑全流程）"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

from src.data.persistence import load_pickle
from src.data.smote import apply_smote_to_minority_classes, get_minority_class_ids
from src.models.mlp import train_mlp_multiclass
from src.models.persistence import save_torch_model

torch.set_num_threads(4)

# Load
X_train = load_pickle(PROJECT_ROOT / "outputs/processed/X_train.pkl")
X_test = load_pickle(PROJECT_ROOT / "outputs/processed/X_test.pkl")
y_train_multi = load_pickle(PROJECT_ROOT / "outputs/processed/y_train_multi.pkl")
y_test_multi = load_pickle(PROJECT_ROOT / "outputs/processed/y_test_multi.pkl")

X_tr, X_val, y_tr_multi, y_val_multi = train_test_split(
    X_train, y_train_multi, test_size=0.2, random_state=42, stratify=y_train_multi
)

# SMOTE
print("Running SMOTE...")
t0 = time.time()
minority_ids_all = get_minority_class_ids(y_tr_multi, threshold=1000)
minority_ids = [cls for cls in minority_ids_all if (y_tr_multi == cls).sum() >= 3]
skipped = [cls for cls in minority_ids_all if cls not in minority_ids]
print(f"  All minority: {len(minority_ids_all)} classes, filtered: {len(minority_ids)}, skipped: {skipped}")
X_smote, y_smote = apply_smote_to_minority_classes(
    X_tr, y_tr_multi, target_classes=minority_ids, k_neighbors=2
)
print(f"  SMOTE: {X_tr.shape[0]} -> {X_smote.shape[0]} samples, {time.time() - t0:.1f}s")

# Train MLP multiclass on SMOTE data
print("\nTraining MLP multiclass on SMOTE data (30 epochs)...")
t0 = time.time()
result = train_mlp_multiclass(
    X_smote, y_smote, X_val, y_val_multi,
    num_classes=23, epochs=30, verbose=False,
)
print(f"  Training: {time.time() - t0:.1f}s")

# Evaluate
preds = result.model.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score
full_acc = float(accuracy_score(y_test_multi, preds))
f1_macro = float(f1_score(y_test_multi, preds, average="macro", zero_division=0))

# Known class accuracy
from collections import Counter
class_counts = Counter(y_test_multi.tolist())
rare = {cls for cls, count in class_counts.items() if count < 5}
mask = ~np.isin(y_test_multi, list(rare))
known_acc = float(accuracy_score(y_test_multi[mask], preds[mask])) if mask.any() else full_acc

# Per-class recall
per_class_recall = recall_score(
    y_test_multi, preds, average=None, labels=list(range(23)), zero_division=0
)

print(f"\n  full_accuracy: {full_acc:.4f}")
print(f"  known_class_accuracy: {known_acc:.4f}")
print(f"  f1_macro: {f1_macro:.4f}")
print(f"  Minority class recall (SMOTE):")
for cls in minority_ids:
    print(f"    class {cls}: {per_class_recall[cls]:.4f}")

# Save model
save_torch_model(result.model, PROJECT_ROOT / "outputs/models/mlp_multiclass_smote.pt")
print(f"\n[done] Saved mlp_multiclass_smote.pt")

# Update metrics JSON
metrics_path = PROJECT_ROOT / "outputs/metrics_m4.json"
with open(metrics_path) as f:
    metrics = json.load(f)

metrics["mlp_multiclass_smote"] = {
    "full_accuracy": full_acc,
    "known_class_accuracy": known_acc,
    "f1_macro": f1_macro,
    "per_class_recall": {int(i): float(per_class_recall[i]) for i in range(23)},
    "minority_classes": minority_ids,
    "skipped_classes": skipped,
    "n_smote_samples": int(X_smote.shape[0]),
}

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2, default=str)
print(f"[done] Updated {metrics_path}")
