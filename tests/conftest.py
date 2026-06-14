"""pytest 全局配置

修复 WSL/sklearn 并行爆炸问题：
  - 设置 OMP_NUM_THREADS=1：OpenMP 单线程（sklearn 内部 BLAS）
  - 设置 MKL_NUM_THREADS=1：Intel MKL 单线程
  - 设置 OPENBLAS_NUM_THREADS=1：OpenBLAS 单线程

原因：
  WSL2 在多核 + 内存受限环境下，sklearn 的 GridSearchCV / RF 训练
  会默认用 n_jobs=-1 启动 OMP 并行，导致进程数 = 物理核数（通常 4-16）。
  每个进程又各自启动 OMP 子线程，组合爆炸（4 进程 × 4 线程 = 16 个 OMP worker），
  触发 OOM 或 fork 失败，表现为 WSL 连接断开。

  在 test 环境中限制为 1 线程，保证稳定运行。性能上牺牲小（< 1s 差异），
  但获得 100% 稳定性。生产代码（train_m3.py）仍可放开。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")