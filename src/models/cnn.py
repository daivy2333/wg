"""1D CNN 神经网络模型（M4 任务 4.4）

架构：Conv1d(1→16, kernel=3) → ReLU → MaxPool1d(2) → Flatten → Linear → output
tabular 数据 (20 维) 视为长度 20 的 1D 序列。

负责人：C同学（神经网络方向）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.mlp import NSLKDDDataset, make_dataloader


class CNN1DClassifier(nn.Module):
    """1D 卷积神经网络分类器

    架构：
        Input (batch, 1, 20)  ← tabular 20 维
        → Conv1d(1→16, kernel=3, padding=1)  ← (batch, 16, 20)
        → ReLU
        → MaxPool1d(2)                       ← (batch, 16, 10)
        → Flatten                            ← (batch, 160)
        → Linear(160 → output_dim)
    """

    def __init__(
        self,
        input_length: int = 20,
        output_dim: int = 2,
        conv_channels: tuple[int, ...] = (16,),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if input_length != 20:
            # 任务范围限定 NSL-KDD Top-20 特征
            raise ValueError(
                f"CNN1DClassifier 需要 input_length=20（NSL-KDD Top-K 特征），当前为 {input_length}"
            )

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in conv_channels:
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(2))
            in_channels = out_channels

        # 计算 flatten 后的维度（20 → pool → 10 → pool → 5，取决于 conv 层数）
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            for layer in layers:
                dummy = layer(dummy)
            flatten_dim = dummy.numel()

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(flatten_dim, output_dim)
        self.input_length = input_length
        self.output_dim = output_dim
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 shape: (batch, 1, input_length=20)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 20) → (batch, 1, 20)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        if isinstance(x, pd.DataFrame):
            x = x.values
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(1)
            logits = self.forward(x_t)
            return torch.argmax(logits, dim=1).cpu().numpy()


@dataclass
class TrainResult:
    model: CNN1DClassifier
    val_metrics: dict[str, float]
    train_history: dict[str, list[float]]


def train_cnn_binary(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    *,
    conv_channels: tuple[int, ...] = (16,),
    kernel_size: int = 3,
    dropout: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 50,
    patience: int = 5,
    random_state: int = 42,
    verbose: bool = False,
) -> TrainResult:
    """训练 1D CNN 二分类模型。"""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    model = CNN1DClassifier(
        input_length=20,
        output_dim=2,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        dropout=dropout,
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
        all_probs: list[float] = []
        all_labels: list[int] = []
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * len(xb)
                probs = F.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(yb.cpu().numpy().tolist())
                n_val += len(xb)

        train_loss = train_loss_sum / max(n_train, 1)
        val_loss = val_loss_sum / max(n_val, 1)
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        val_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_true, y_prob)),
        }
        val_f1 = val_metrics["f1"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_auc"].append(val_metrics["auc"])

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(
                f"epoch {epoch+1:3d}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_f1={val_f1:.4f}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1} (best f1={best_val_f1:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # 重新评估
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    final_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }
    return TrainResult(model=model, val_metrics=final_metrics, train_history=history)
