"""pytest 全局配置

四层防御（防 WSL 断开/OOM）：
  1. OMP/MKL/OPENBLAS 线程数 = 1
  2. torch CPU 线程 = 1 + interop 线程 = 1
  3. CUDA_VISIBLE_DEVICES=""（测试禁用 GPU）
  4. autouse fixture：每个测试后强制 gc.collect() + torch.cuda.empty_cache()

为什么需要第四层？
  - WSL2 内存通常 8GB，pytest 收集 14 文件 × 130 测试
  - PyTorch 即使禁用 CUDA，模型对象仍驻留 heap
  - 不显式清理 → 内存累积 → OOM → exit 137 → WSL 断开
"""
import gc
import os

# 第一层：环境变量（必须在 import 任何 ML 库之前）
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# 第二层：torch CPU 线程
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# 第四层：autouse fixture，每个测试后清理内存
import pytest


@pytest.fixture(autouse=True)
def _cleanup_memory_after_test():
    """每个测试后强制释放内存（防 WSL OOM 断开）。"""
    yield
    # 测试结束后清理
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
