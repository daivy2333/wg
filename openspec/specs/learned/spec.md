# Learned Spec

> Version: 1.0.0 | Last Updated: 2026-06-14

## Purpose

记录项目开发过程中学到的知识，避免重复探索，加速问题解决。

## Requirements

### Requirement: API 路径记录

项目中使用的关键 API 路径 SHALL 记录，包含用途和使用示例。

#### Scenario: 发现新 API

- **WHEN** 开发者发现或使用了新的 API 端点
- **THEN** 必须记录到 learned/spec.md，包含：API 路径、用途、请求/响应格式

### Requirement: 踩坑经验记录

遇到的技术陷阱和解决方案 SHALL 记录，防止重复踩坑。

#### Scenario: 解决棘手问题

- **WHEN** 开发者花费大量时间解决了一个技术问题
- **THEN** 必须记录踩坑档案，包含：症状、根因、解决方案、预防措施

### Requirement: 技巧模式记录

有效的开发技巧和模式 SHALL 记录，促进知识共享。

#### Scenario: 发现高效做法

- **WHEN** 开发者发现了一种高效的开发技巧或模式
- **THEN** 必须记录到技巧模式区，包含：技巧名称、适用场景、使用方法

### Requirement: 文件速查表

关键文件和目录的位置 SHALL 记录，加速代码导航。

#### Scenario: 定位关键文件

- **WHEN** 开发者频繁访问某些文件或目录
- **THEN** 必须记录到文件速查表，包含：文件路径、用途、关键内容

## 踩坑档案

### 踩坑-001: which conda 无输出但 conda 可用

- **症状**: 运行 `which conda` 无输出，但 `conda activate` 可正常工作
- **根因**: conda 通过 shell 函数初始化（`eval "$(conda shell.bash hook)"`），而非作为独立可执行文件安装到 PATH，因此 `which` 找不到
- **解决方案**: 直接使用 `conda activate` 激活环境，不依赖 `which` 检测
- **预防措施**: 检测 conda 可用性时，用 `conda --version` 或 `type conda` 替代 `which conda`
- **结论**: 项目使用 conda（miniconda）管理 Python 环境，后续所有 Python 命令应在 conda 环境中执行

### 踩坑-002: bash 子 shell 中 conda 不可用

- **症状**: Claude Code 的 Bash 工具（子 shell）中 `conda info --envs` 报 `conda: command not found`，但用户终端中 conda 正常工作
- **根因**: conda 通过 `~/.bashrc` 中的 shell 函数初始化，Bash 工具的子 shell 不会 source `~/.bashrc`，因此 conda 函数和路径均不可用
- **解决方案**: 用户在终端中执行 conda 相关操作；Claude Code 中直接使用 `/home/daivy/miniconda3/bin/python` 运行脚本
- **预防措施**: 需要安装包或管理 conda 环境时，提示用户在终端执行
- **结论**: Claude Code 中执行 Python 代码应使用 `/home/daivy/miniconda3/bin/python`（Python 3.13.5），而非 `/usr/bin/python3`（Python 3.10.12）

## 文件速查表

| 文件路径 | 用途 | 关键内容 |
|----------|------|----------|
| `readme` | 项目说明文档 | 技术栈、数据集说明、分工建议、实验流程 |
| `dataset/KDDTrain+.txt` | NSL-KDD 训练集 | 125,973 条，41 特征 + 标签 + 难度 |
| `dataset/KDDTest+.txt` | NSL-KDD 测试集 | 22,544 条，41 特征 + 标签 + 难度 |
| `openspec/config.yaml` | OpenSpec 项目配置 | 技术栈、规则、上下文 |
| `/home/daivy/miniconda3/bin/python` | conda 环境 Python | Python 3.13.5，所有依赖已安装 |
