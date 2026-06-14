# paper-cjc-supplementary

> 英文背景、算法伪代码、附录等补充模块
> Version: 1.0.0 | Last Updated: 2026-06-14

## ADDED Requirements

### Requirement: 英文论文背景
论文 SHALL 在参考文献后包含 `background` 环境，用英文撰写约 400 词的论文背景介绍，涵盖研究领域、课题意义、团队成果与本文贡献。

#### Scenario: background 环境正确渲染
- **WHEN** 编译后查看 PDF 末尾（参考文献之后）
- **THEN** `\begin{background}...\end{background}` 内容以英文小五号 Times New Roman 字体显示
- **AND** 内容包含：研究领域概述、国际研究现状、本文解决程度、课题所属项目、团队以往成果、本文在课题中的定位
- **AND** 内容约 350–450 个英文单词

### Requirement: 算法伪代码
论文 SHALL 在附录或正文适当位置包含至少一个 `algorithm` 或 `procedure` 环境，展示关键实验流程的伪代码（如训练-评估循环或特征工程管线）。

#### Scenario: 算法伪代码正确渲染
- **WHEN** 编译后查看 PDF
- **THEN** algorithm/procedure 环境包含标题（caption）
- **AND** 伪代码使用 `\STATE`、`\IF`/`\ELSE`、`\FOR`/`\WHILE` 等标准关键字
- **AND** 关键字全部大写（IF、THEN、ELSE、FOR、WHILE）
- **AND** 变量和函数名使用斜体

#### Scenario: 至少一个伪代码块
- **WHEN** 统计论文中的 algorithm 和 procedure 环境
- **THEN** 至少包含 1 个伪代码块
- **AND** 伪代码块内容与实验流程相关（如模型训练循环、评估流程、特征工程管线）

### Requirement: 附录
论文 SHALL 在 `\appendix` 声明后包含至少一个附录章节，放置补充表格、详细参数或推导过程。

#### Scenario: 附录正确渲染
- **WHEN** 编译后查看 PDF
- **THEN** `\appendix` 后的 `\section{}` 标题自动编号为字母（如 A, B, C）
- **AND** 附录内容使用小五号宋体

#### Scenario: 附录包含补充材料
- **WHEN** 查看附录内容
- **THEN** 至少包含一项补充内容（如：模型超参数完整列表、SMOTE 详细实验结果表、特征完整排名、额外对比图表）

### Requirement: 致谢
论文 SHALL 包含 `acknowledgments` 环境，致谢研究过程中提供帮助的个人或机构。

#### Scenario: acknowledgments 环境存在
- **WHEN** 查看 PDF 致谢部分
- **THEN** 如果填写了 grants 字段且内容非空，应包含致谢段落
- **AND** 致谢内容置于参考文献之前
