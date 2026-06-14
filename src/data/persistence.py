"""数据持久化模块（M2 任务 2.9）

pickle 序列化/反序列化 + M2 数据集整合保存
"""
from pathlib import Path
from typing import Literal

import pandas as pd
import pickle


def save_pickle(obj, path: str | Path) -> None:
    """保存任意对象到 pickle 文件。

    自动创建中间目录。

    Args:
        obj: 任意 pickle 可序列化对象
        path: 目标文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path):
    """从 pickle 文件加载对象。

    Args:
        path: pickle 文件路径

    Returns:
        还原后的对象

    Raises:
        FileNotFoundError: 当 path 不存在时
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pickle 文件不存在：{path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_m2_datasets(
    X_train,
    X_test,
    y_train,
    y_test,
    output_dir: str | Path,
    task: Literal["binary", "multiclass"] = "binary",
) -> None:
    """保存 M2 数据集到指定目录（二分类 + 多分类双套）。

    Args:
        X_train: 训练特征（DataFrame 或 ndarray）
        X_test: 测试特征
        y_train: 训练标签
        y_test: 测试标签
        output_dir: 输出目录（不存在则自动创建）
        task: 'binary' → 文件名无后缀；'multiclass' → 文件名加 _multi 后缀

    Raises:
        ValueError: 当 task 不是 'binary' 或 'multiclass' 时
    """
    if task not in ("binary", "multiclass"):
        raise ValueError(f"未知 task: {task!r}（仅支持 'binary' 或 'multiclass'）")

    suffix = "" if task == "binary" else "_multi"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_pickle(X_train, output_dir / f"X_train{suffix}.pkl")
    save_pickle(X_test, output_dir / f"X_test{suffix}.pkl")
    save_pickle(y_train, output_dir / f"y_train{suffix}.pkl")
    save_pickle(y_test, output_dir / f"y_test{suffix}.pkl")