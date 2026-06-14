# paper-cjc-template

> CJC 模板主文件与编译流程
> Version: 1.0.0 | Last Updated: 2026-06-14

## ADDED Requirements

### Requirement: CJC 模板主文件编译
系统 SHALL 提供一个基于 `cjc.cls` 的 LaTeX 主文件（`paper/cjc-main.tex`），通过 `latexmk -xelatex` 成功编译生成 PDF。

#### Scenario: 使用 latexmk 成功编译
- **WHEN** 用户在 `paper/` 目录执行 `latexmk -xelatex cjc-main.tex`
- **THEN** 编译过程退出码为 0，生成 `cjc-main.pdf`
- **AND** PDF 包含封面（中英文标题、作者、单位）、中英文摘要与关键词、目录

#### Scenario: 使用 make 成功编译
- **WHEN** 用户在 `paper/` 目录执行 `make MAIN=cjc-main`
- **THEN** 编译流程自动完成（xelatex + bibtex 多轮），生成 `cjc-main.pdf`

### Requirement: 中英文双语元数据
`\classsetup{}` 配置块 SHALL 包含中英文双语标题（title/title*）、摘要（abstract/abstract*）、关键词（keywords/keywords*），中文为主、英文为对应翻译。

#### Scenario: classsetup 包含完整双语元数据
- **WHEN** 读取 `cjc-main.tex` 的 `\classsetup{}` 块
- **THEN** title 字段为中文论文标题
- **AND** title* 字段为英文对应标题
- **AND** abstract 字段包含中文摘要（约 200–300 字）
- **AND** abstract* 字段包含英文摘要（约 150–200 词）
- **AND** 中英文关键词各 5–7 个且一一对应

### Requirement: 作者信息占位符
`\classsetup{}` 中的 authors 块 SHALL 包含 3 位作者，每位作者含中英文姓名、单位引用（affiliations）、中英文简介（biography/biography*）、email，信息以 `[TODO]` 占位符标记待手动填写。

#### Scenario: 作者信息使用占位符
- **WHEN** 读取 `cjc-main.tex` 的 authors 配置
- **THEN** 每位作者的 name/name* 字段包含 `[TODO: 作者姓名]` 或中文占位文本
- **AND** biography/biography* 字段包含格式模板文本
- **AND** email 字段为 `[TODO: email]`
- **AND** 至少一位作者标记 `corresponding = true`

### Requirement: 基金致谢占位符
`\classsetup{}` 中的 grants 字段 SHALL 包含占位符，提示填写基金项目信息或注明无基金资助。

#### Scenario: 基金信息占位符
- **WHEN** 读取 grants 字段
- **THEN** 内容为 `[TODO: 请填写基金项目完整名称及编号，如无基金资助请删除此行]`

### Requirement: CJC 参考文献格式
论文 SHALL 使用 `cjc.bst` 作为参考文献格式文件，编译时自动按正文引用顺序排列参考文献。

#### Scenario: 参考文献按引用顺序排列
- **WHEN** 编译完成后查看 PDF 参考文献列表
- **THEN** 参考文献编号为 [1], [2], [3]... 按正文首次出现顺序排列
- **AND** 条目包含 author, title, journal/booktitle, year, volume, pages 等完整字段

### Requirement: 章节自动编号与目录
`\maketitle` 后 SHALL 自动生成目录（含章节编号），`\section`/`\subsection` 自动编号并与目录链接一致。

#### Scenario: 目录包含全部章节
- **WHEN** 编译后查看 PDF 目录页
- **THEN** 目录列出 6 章一级标题及对应页码
- **AND** 各章 `\section` 自动编号为 1–6
- **AND** 子节 `\subsection` 自动编号为 X.1, X.2...
