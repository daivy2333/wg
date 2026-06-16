"""MLP (Multi-Layer Perceptron) 神经网络模型（M4 任务 4.1/4.2/4.3）

基于 PyTorch 自建的多层感知器，用于 NSL-KDD 入侵检测：
  - 二分类：normal vs anomaly
  - 多分类：23 类攻击类型

负责人：C同学（神经网络方向）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 1. MLP 模型定义
# ============================================================


class MLPClassifier(nn.Module):
    """多层感知器分类器（PyTorch nn.Module）

    架构：Input → Dense(hidden_dims[0]) → ReLU → Dropout
                       → ... → Dense(hidden_dims[-1]) → ReLU → Dropout
                       → Linear(output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation not in ("relu", "tanh"):
            raise ValueError(f"activation 必须是 'relu' 或 'tanh'，当前为 {activation!r}")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：x.shape = (batch, input_dim) → logits.shape = (batch, output_dim)"""
        return self.net(x)

    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """预测接口（无梯度）。返回 class index 数组。"""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            x_t = self._to_tensor(x).to(device)
            logits = self.forward(x_t)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """预测概率（二分类返回 2 维，多分类返回 num_classes 维）。"""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            x_t = self._to_tensor(x).to(device)
            logits = self.forward(x_t)
            return F.softmax(logits, dim=1).cpu().numpy()

    @staticmethod
    def _to_tensor(x: np.ndarray | pd.DataFrame) -> torch.Tensor:
        if isinstance(x, pd.DataFrame):
            x = x.values
        return torch.as_tensor(x, dtype=torch.float32)


# ============================================================
# 2. Dataset / DataLoader
# ============================================================


class NSLKDDDataset(Dataset):
    """NSL-KDD 训练样本的 PyTorch Dataset 包装。

    X.shape = (N, input_dim)
    y.shape = (N,) — 二分类 0/1 或多分类 0~22
    """

    def __init__(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> None:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_dataloader(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """构造 DataLoader 的便捷函数。"""
    dataset = NSLKDDDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ============================================================
# 3. 训练结果 / 指标
# ============================================================


@dataclass
class TrainResult:
    """训练结果汇总。"""

    model: MLPClassifier
    val_metrics: dict[str, float]
    train_history: dict[str, list[float]]  # epoch -> {train_loss, val_loss, val_f1, ...}


# ============================================================
# 4. 训练函数（二分类）
# ============================================================


def train_mlp_binary(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    *,
    hidden_dims: tuple[int, ...] = (128, 64),
    dropout: float = 0.3,
    activation: str = "relu",
    lr: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 50,
    patience: int = 5,
    random_state: int = 42,
    verbose: bool = False,
) -> TrainResult:
    """训练 MLP 二分类模型。

    特性：
      - 早停（patience=5）：验证 f1 连续 5 epoch 不提升则停止
      - ReduceLROnPlateau：验证 loss 3 epoch 不降则学习率减半
      - 设备：自动检测 CUDA / CPU

    Returns:
        TrainResult 包含 model + val_metrics + train_history
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    model = MLPClassifier(
        input_dim=X_train.shape[1] if hasattr(X_train, "shape") else X_train[0].shape[0],
        output_dim=2,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    no_improve = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_accuracy": [],
        "val_auc": [],
    }

    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
            n_train += len(xb)

        # 验证
        model.eval()
        val_loss_sum = 0.0
        all_preds: list[int] = []
        all_probs: list[float] = []
        all_labels: list[int] = []
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * len(xb)
                probs = F.softmax(logits, dim=1)[:, 1]  # anomaly 类的概率
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(yb.cpu().numpy().tolist())
                n_val += len(xb)

        train_loss = train_loss_sum / max(n_train, 1)
        val_loss = val_loss_sum / max(n_val, 1)
        val_metrics = _compute_binary_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
        val_f1 = val_metrics["f1"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])

        # 早停
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        scheduler.step(val_loss)

        if verbose:
            print(
                f"epoch {epoch + 1:3d}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_f1={val_f1:.4f} val_auc={val_metrics['auc']:.4f}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1} (best f1={best_val_f1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # 重新在验证集上评估（确保返回的是 best epoch 的指标）
    final_metrics = _evaluate_binary(model, val_loader, device)
    return TrainResult(model=model, val_metrics=final_metrics, train_history=history)


def _evaluate_binary(model: MLPClassifier, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """在给定 DataLoader 上计算二分类指标。"""
    model.eval()
    all_preds: list[int] = []
    all_probs: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    return _compute_binary_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """计算 accuracy / precision / recall / f1 / auc。"""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else 0.0,
    }


# ============================================================
# 5. 训练函数（多分类）
# ============================================================


# 训练集中没有的 16 种攻击（M2 EDA 已识别）
UNSEEN_ATTACKS_IN_TEST = (
    "apache2",
    "httptunnel",
    "mailbomb",
    "mscan",
    "named",
    "processtable",
    "saint",
    "sendmail",
    "snmpgetattack",
    "snmpguess",
    "sqlattack",
    "udpstorm",
    "worm",
    "xterm",
    "xlock",
    "xsnoop",
)


def train_mlp_multiclass(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    *,
    num_classes: int = 5,
    hidden_dims: tuple[int, ...] = (128, 128),
    dropout: float = 0.3,
    class_weight: str | dict[int, float] | None = "balanced",
    lr: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 50,
    patience: int = 5,
    random_state: int = 42,
    verbose: bool = False,
) -> TrainResult:
    """训练 MLP 多分类模型（23 类）。

    评估指标：
      - accuracy
      - f1_macro
      - per_class_f1（每类 F1 字典）
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1] if hasattr(X_train, "shape") else X_train[0].shape[0]
    model = MLPClassifier(
        input_dim=input_dim,
        output_dim=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    # class_weight 处理
    weight = None
    if class_weight == "balanced":
        y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
        class_counts = np.bincount(y_train_arr, minlength=num_classes)
        # 过滤 0 计数（避免除零）
        class_counts = np.where(class_counts == 0, 1, class_counts)
        weight = torch.tensor(len(y_train_arr) / (num_classes * class_counts), dtype=torch.float32).to(device)
    elif isinstance(class_weight, dict):
        weight = torch.tensor([class_weight.get(i, 1.0) for i in range(num_classes)], dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_val_f1 = -1.0
    best_state = None
    no_improve = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_f1_macro": [],
        "val_accuracy": [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
            n_train += len(xb)

        model.eval()
        val_loss_sum = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * len(xb)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(yb.cpu().numpy().tolist())
                n_val += len(xb)

        train_loss = train_loss_sum / max(n_train, 1)
        val_loss = val_loss_sum / max(n_val, 1)
        from sklearn.metrics import f1_score

        val_f1_macro = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        val_accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1_macro"].append(val_f1_macro)
        history["val_accuracy"].append(val_accuracy)

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(
                f"epoch {epoch + 1:3d}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_f1_macro={val_f1_macro:.4f} val_acc={val_accuracy:.4f}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1} (best f1_macro={best_val_f1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    final_metrics = _evaluate_multiclass(model, val_loader, device, num_classes)
    return TrainResult(model=model, val_metrics=final_metrics, train_history=history)


def _evaluate_multiclass(
    model: MLPClassifier, loader: DataLoader, device: torch.device, num_classes: int
) -> dict[str, float]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    from sklearn.metrics import accuracy_score, f1_score

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {int(i): float(per_class_f1[i]) for i in range(num_classes)},
    }


def compute_known_class_accuracy(
    model: MLPClassifier,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    unseen_label_ids: Sequence[int] = (),  # 默认空：23 类训练中所有类都"已知"
) -> float:
    """计算剔除 unseen 类后的已知类准确率。

    Args:
        unseen_label_ids: 视为 unseen 的标签 id 列表（这些类在训练中完全没出现过）
            对于 NSL-KDD 多分类，如果测试集出现训练中 16 种 unseen 攻击，
            它们在 y_test 中仍会被预测为某个训练类（最相似的），导致 acc 偏低。
            剔除这些样本后计算"已知类 acc"更公平。

    Note:
        训练集中没有的 16 种攻击（UNSEEN_ATTACKS_IN_TEST）在训练数据中根本不存在，
        因此 y_train 中也不会有它们的 id。调用者需要根据 test 集的真实分布
        判断哪些标签是 unseen（如果有）。
    """
    model.eval()
    device = next(model.parameters()).device
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    if not unseen_label_ids:
        # 全部视为已知
        preds = _predict_on_device(model, X_test)
        return float(np.mean(preds == y_test))

    # 剔除 unseen 样本
    y_test_arr = np.asarray(y_test)
    mask = ~np.isin(y_test_arr, list(unseen_label_ids))
    if not mask.any():
        return 0.0
    X_known = X_test[mask]
    y_known = y_test_arr[mask]
    preds = _predict_on_device(model, X_known)
    return float(np.mean(preds == y_known))


def _predict_on_device(model: MLPClassifier, X: np.ndarray) -> np.ndarray:
    """在模型所在设备上预测（避免 device mismatch）。"""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        logits = model(x_t)
        return torch.argmax(logits, dim=1).cpu().numpy()


# ============================================================
# 6. 参数调优（网格搜索）
# ============================================================


DEFAULT_PARAM_GRID = {
    "hidden_dims": [(64,), (128, 64), (256, 128, 64)],
    "activation": ["relu", "tanh"],
    "dropout": [0.3],
    "lr": [1e-3],
}


def grid_search_mlp(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    param_grid: dict | None = None,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    patience: int = 3,
    random_state: int = 42,
    cv_folds: int = 3,
    verbose: bool = False,
) -> tuple["MLPClassifier", dict, dict]:
    """MLP 二分类网格搜索调优。

    Args:
        X_train, y_train, X_val, y_val: 训练/验证集
        param_grid: 网格字典，默认 DEFAULT_PARAM_GRID（4×2×2×2 = 32 组合）
        epochs: 每个组合的训练 epoch 数（调优时用较少 epoch 加速）
        batch_size, patience, random_state: 训练超参
        cv_folds: CV 折数（默认 3 折，5 折更慢）
        verbose: 是否打印每个组合的进度

    Returns:
        (best_model, best_params, summary) — 最佳模型 + 最佳参数 + 全部结果摘要
        summary 含每个组合的 cv_f1 + val_f1（双数字）
    """
    from itertools import product
    from sklearn.model_selection import StratifiedKFold

    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    # 生成所有参数组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    if verbose:
        print(f"网格搜索：{len(combinations)} 组合 × {cv_folds} 折 CV")

    # K 折划分（用随机种子保证可复现）
    if isinstance(X_train, pd.DataFrame):
        X_train_arr = X_train.values
    else:
        X_train_arr = np.asarray(X_train)
    if isinstance(y_train, pd.Series):
        y_train_arr = y_train.values
    else:
        y_train_arr = np.asarray(y_train)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    folds = list(skf.split(X_train_arr, y_train_arr))

    summary: dict[str, dict] = {}

    for idx, params in enumerate(combinations):
        cv_f1_scores: list[float] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            X_tr_fold = X_train_arr[tr_idx]
            y_tr_fold = y_train_arr[tr_idx]
            X_val_fold = X_train_arr[val_idx]
            y_val_fold = y_train_arr[val_idx]

            result = train_mlp_binary(
                X_tr_fold,
                y_tr_fold,
                X_val_fold,
                y_val_fold,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                random_state=random_state,
                hidden_dims=params["hidden_dims"],
                activation=params["activation"],
                dropout=params["dropout"],
                lr=params["lr"],
                verbose=False,
            )
            cv_f1_scores.append(result.val_metrics["f1"])

        cv_f1 = float(np.mean(cv_f1_scores))

        # 在完整 train+val 上训练，记录最终 val f1
        result_full = train_mlp_binary(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            random_state=random_state,
            hidden_dims=params["hidden_dims"],
            activation=params["activation"],
            dropout=params["dropout"],
            lr=params["lr"],
            verbose=False,
        )
        val_f1 = result_full.val_metrics["f1"]

        param_key = _params_to_key(params)
        summary[param_key] = {
            "params": params,
            "cv_f1": cv_f1,
            "val_f1": val_f1,
        }

        if verbose:
            print(
                f"[{idx + 1}/{len(combinations)}] {param_key} "
                f"cv_f1={cv_f1:.4f} val_f1={val_f1:.4f}"
            )

    # 选 val_f1 最高的组合
    best_key = max(summary, key=lambda k: summary[k]["val_f1"])
    best_params = summary[best_key]["params"]

    # 用最佳参数在完整数据上重新训练，返回 best model
    best_result = train_mlp_binary(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs * 2,  # 调优时 epochs 较少，最终用更多 epoch 训练
        batch_size=batch_size,
        patience=patience * 2,
        random_state=random_state,
        hidden_dims=best_params["hidden_dims"],
        activation=best_params["activation"],
        dropout=best_params["dropout"],
        lr=best_params["lr"],
        verbose=False,
    )

    if verbose:
        print(
            f"\n最佳参数: {best_key}\n"
            f"  cv_f1={summary[best_key]['cv_f1']:.4f}\n"
            f"  val_f1={summary[best_key]['val_f1']:.4f}\n"
            f"  重新训练后 val_f1={best_result.val_metrics['f1']:.4f}"
        )

    return best_result.model, best_params, summary


def _params_to_key(params: dict) -> str:
    """将参数 dict 转为可读 key。"""
    return (
        f"hidden={params['hidden_dims']}_act={params['activation']}"
        f"_drop={params['dropout']}_lr={params['lr']}"
    )
