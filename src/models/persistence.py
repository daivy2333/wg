"""模型持久化模块

支持两种后端（职责分明，遵循 ADR-005）：
  - joblib: sklearn 模型（M3）
  - torch.save: PyTorch 模型（M4 新增）

负责人：C同学（M4 扩展）
"""
from pathlib import Path
from typing import Any

import joblib
import torch
import torch.nn as nn


# ============================================================
# joblib 后端（sklearn，M3 沿用）
# ============================================================


def save_model(model, path: str | Path) -> None:
    """保存 sklearn 模型到 joblib 文件。

    joblib 对 sklearn 模型比 pickle 更高效（内部 numpy 二进制 + 压缩）。

    Args:
        model: 任意 sklearn 模型
        path: 目标文件路径（.joblib），自动创建中间目录
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    """加载 joblib 模型。

    Args:
        path: joblib 文件路径

    Returns:
        还原的 sklearn 模型

    Raises:
        FileNotFoundError: 当 path 不存在
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在：{path}")
    return joblib.load(path)


def save_best_models(
    dt_binary,
    rf_binary,
    dt_multiclass,
    rf_multiclass,
    output_dir: str | Path,
) -> None:
    """M3 最佳模型批量保存：DT/RF × 二分类/多分类 = 4 个文件。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_model(dt_binary, output_dir / "dt_binary_best.joblib")
    save_model(rf_binary, output_dir / "rf_binary_best.joblib")
    save_model(dt_multiclass, output_dir / "dt_multiclass_best.joblib")
    save_model(rf_multiclass, output_dir / "rf_multiclass_best.joblib")


# ============================================================
# torch.save 后端（PyTorch，M4 新增）
# ============================================================


def save_torch_model(model: nn.Module, path: str | Path) -> None:
    """保存 PyTorch 模型 state_dict 到 .pt 文件。

    仅保存 state_dict 而非整个 model 对象（PyTorch 官方推荐）：
      - 体积更小
      - 不绑定 Python 类路径（避免 pickle 风险）
      - 跨版本兼容性更好

    Args:
        model: 任意 PyTorch nn.Module 实例
        path: 目标文件路径（.pt），自动创建中间目录
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_torch_model(
    model_class: type[nn.Module],
    path: str | Path,
    **model_kwargs: Any,
) -> nn.Module:
    """加载 PyTorch 模型 state_dict 并还原模型实例。

    Args:
        model_class: 模型类（如 MLPClassifier）
        path: .pt 文件路径
        **model_kwargs: 传给 model_class.__init__ 的关键字参数（如 input_dim=20, output_dim=2）

    Returns:
        还原的 PyTorch 模型实例（state_dict 已加载）

    Raises:
        FileNotFoundError: 当 path 不存在
        RuntimeError: 当 state_dict 与 model_kwargs 构造的模型维度不匹配
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在：{path}")

    model = model_class(**model_kwargs)
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"加载 state_dict 失败：可能 model_kwargs 不匹配（实际尺寸 vs 期望尺寸）。"
            f"原始错误：{e}"
        ) from e
    model.eval()
    return model


def save_best_nn_models(
    output_dir: str | Path,
    *,
    mlp_binary=None,
    mlp_binary_tuned=None,
    mlp_multiclass=None,
    cnn_binary=None,
    lstm_binary=None,
    mlp_multiclass_smote=None,
) -> list[str]:
    """M4 神经网络模型批量保存。

    Args:
        output_dir: 输出目录（不存在则自动创建）
        mlp_binary, mlp_binary_tuned, mlp_multiclass, cnn_binary, lstm_binary, mlp_multiclass_smote:
            各 PyTorch 模型实例（None 表示未训练，跳过）

    Returns:
        已保存的文件路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    model_map = {
        "mlp_binary_best.pt": mlp_binary,
        "mlp_binary_tuned.pt": mlp_binary_tuned,
        "mlp_multiclass_best.pt": mlp_multiclass,
        "cnn_binary_best.pt": cnn_binary,
        "lstm_binary_best.pt": lstm_binary,
        "mlp_multiclass_smote.pt": mlp_multiclass_smote,
    }
    for filename, model in model_map.items():
        if model is not None:
            save_torch_model(model, output_dir / filename)
            saved_paths.append(str(output_dir / filename))
        else:
            print(f"[skip] {filename} 未提供，跳过")
    return saved_paths
