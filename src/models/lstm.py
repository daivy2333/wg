"""LSTM 神经网络模型（M4 任务 4.5）

架构：LSTM(input_size=20, hidden=32) → Linear → output
tabular 样本的 20 维特征视为长度 1 的序列（每个特征作为独立时间步）。

诚实记录局限：NSL-KDD 样本独立无时序关系，LSTM 优势有限，仅作为论文对比维度。

负责人：C同学（神经网络方向）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mlp import make_dataloader


class LSTMClassifier(nn.Module):
    """LSTM 分类器

    架构：
        Input (batch, seq_len=1, features=20)  ← tabular 20 维
        → LSTM(input_size=20, hidden_size=32, batch_first=True)
        → 取最后 hidden state (batch, 32)
        → Linear(32 → output_dim)
    """

    def __init__(
        self,
        input_size: int = 20,
        output_dim: int = 2,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if input_size != 20:
            raise ValueError(
                f"LSTMClassifier 需要 input_size=20（NSL-KDD Top-K 特征），当前为 {input_size}"
            )
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.input_size = input_size
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 shape: (batch, 1, 20) — 序列长度 1，特征 20
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 20) → (batch, 1, 20)
        elif x.dim() == 3 and x.shape[1] != 1:
            raise ValueError(
                f"LSTMClassifier 需要 seq_len=1（tabular 数据），当前输入 shape={tuple(x.shape)}"
            )
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]  # 取最后时间步
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)

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
    model: LSTMClassifier
    val_metrics: dict[str, float]
    train_history: dict[str, list[float]]


def train_lstm_binary(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    *,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 50,
    patience: int = 5,
    random_state: int = 42,
    verbose: bool = False,
) -> TrainResult:
    """训练 LSTM 二分类模型。"""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    model = LSTMClassifier(
        input_size=20,
        output_dim=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
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

    # 重新评估 best 模型
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
