"""NSL-KDD 数据集加载模块

参考：NSL-KDD 官方论文 + KDD Cup 1999 数据集文档
41 特征顺序遵循论文规范（TCP连接基本 → 内容 → 时间 → 主机）
"""
from pathlib import Path
from typing import Final

import pandas as pd


# 数据集根目录（相对于项目根 wg/）
DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "dataset"

# NSL-KDD 41 特征名（按官方论文顺序）
FEATURE_NAMES: Final[list[str]] = [
    # === TCP 连接基本特征 (9) ===
    "duration",            # 连接持续时间
    "protocol_type",       # 协议类型 (tcp/udp/icmp)
    "service",             # 网络服务类型 (http/telnet/...)
    "flag",                # 连接状态标志
    "src_bytes",           # 源→目的字节数
    "dst_bytes",           # 目的→源字节数
    "land",                # 同主机同端口 (0/1)
    "wrong_fragment",      # 错误分片数
    "urgent",              # urgent 包数
    # === TCP 连接内容特征 (13) ===
    "hot",                 # hot 指标数
    "num_failed_logins",   # 失败登录次数
    "logged_in",           # 成功登录 (0/1)
    "num_compromised",     # compromised 条件数
    "root_shell",          # 获得 root shell (0/1)
    "su_attempted",        # su root 尝试 (0/1)
    "num_root",            # root 访问次数
    "num_file_creations",  # 文件创建操作数
    "num_shells",          # shell 提示数
    "num_access_files",    # 访问文件操作数
    "num_outbound_cmds",   # outbound 命令数
    "is_host_login",       # host 登录 (0/1)
    "is_guest_login",      # guest 登录 (0/1)
    # === 基于时间的网络流量统计特征 (9) ===
    "count",               # 2s 内同 host 连接数
    "srv_count",           # 2s 内同 service 连接数
    "serror_rate",         # SYN 错误率
    "srv_serror_rate",     # 同 service SYN 错误率
    "rerror_rate",         # REJ 错误率
    "srv_rerror_rate",     # 同 service REJ 错误率
    "same_srv_rate",       # 同 service 占比
    "diff_srv_rate",       # 不同 service 占比
    "srv_diff_host_rate",  # 不同 host 占比
    # === 基于主机的网络流量统计特征 (10) ===
    "dst_host_count",          # 同目的 host 连接数
    "dst_host_srv_count",      # 同目的 host 同 service 数
    "dst_host_same_srv_rate",  # 同 service 占比
    "dst_host_diff_srv_rate",  # 不同 service 占比
    "dst_host_same_src_port_rate",  # 同 src_port 占比
    "dst_host_srv_diff_host_rate",  # 同 service 不同 host 占比
    "dst_host_serror_rate",    # 目的 host SYN 错误率
    "dst_host_srv_serror_rate",    # 目的 host service SYN 错误率
    "dst_host_rerror_rate",    # 目的 host REJ 错误率
    "dst_host_srv_rerror_rate",    # 目的 host service REJ 错误率
]

# 完整列名：41 特征 + label + difficulty
COLUMN_NAMES: Final[list[str]] = FEATURE_NAMES + ["label", "difficulty"]


def _read_nsl_kdd(file_path: Path) -> pd.DataFrame:
    """内部函数：读取单个 KDD 文件，含错误处理。

    Args:
        file_path: KDDTrain+.txt 或 KDDTest+.txt 的绝对路径

    Returns:
        带正确列名的 DataFrame

    Raises:
        FileNotFoundError: 当文件不存在时
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"数据文件不存在：{file_path}\n"
            f"请检查 dataset/ 目录或解压 archive.zip"
        )

    return pd.read_csv(
        file_path,
        header=None,            # KDD 文件无 header
        names=COLUMN_NAMES,
        encoding="latin-1",     # KDD 学术标准编码
    )


def load_train() -> pd.DataFrame:
    """加载 NSL-KDD 训练集 KDDTrain+.txt

    Returns:
        shape=(125973, 43) 的 DataFrame，包含 41 特征 + label + difficulty
    """
    return _read_nsl_kdd(DATA_DIR / "KDDTrain+.txt")


def load_test() -> pd.DataFrame:
    """加载 NSL-KDD 测试集 KDDTest+.txt

    Returns:
        shape=(22544, 43) 的 DataFrame，包含 41 特征 + label + difficulty
    """
    return _read_nsl_kdd(DATA_DIR / "KDDTest+.txt")