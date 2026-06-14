"""模型持久化模块（M3 任务 3.8）

joblib save/load + M3 最佳模型批量保存
"""
from pathlib import Path

import joblib


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
    """M3 最佳模型批量保存：DT/RF × 二分类/多分类 = 4 个文件。

    Args:
        dt_binary: 二分类 DT 最佳模型
        rf_binary: 二分类 RF 最佳模型
        dt_multiclass: 多分类 DT 最佳模型
        rf_multiclass: 多分类 RF 最佳模型
        output_dir: 输出目录（不存在则自动创建）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_model(dt_binary, output_dir / "dt_binary_best.joblib")
    save_model(rf_binary, output_dir / "rf_binary_best.joblib")
    save_model(dt_multiclass, output_dir / "dt_multiclass_best.joblib")
    save_model(rf_multiclass, output_dir / "rf_multiclass_best.joblib")