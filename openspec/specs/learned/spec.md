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

## 文件速查表

| 文件路径 | 用途 | 关键内容 |
|----------|------|----------|
| `readme` | 项目说明文档 | 技术栈、数据集说明、分工建议、实验流程 |
| `dataset/archive.zip` | NSL-KDD 数据集压缩包 | KDDTrain+.TXT + KDDTest+.TXT（需解压） |
| `openspec/config.yaml` | OpenSpec 项目配置 | 技术栈、规则、上下文 |
