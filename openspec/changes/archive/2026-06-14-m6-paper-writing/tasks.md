## 1. LaTeX 项目骨架

- [ ] 1.1 检查 LaTeX 编译环境（`xelatex --version`），若缺失提示用户安装
- [ ] 1.2 创建 `paper/main.tex`——主文件（ctex 文档类、封面、摘要、`\input` 各章、参考文献）
- [ ] 1.3 创建 `paper/refs.bib`——BibTeX 参考文献（NSL-KDD 论文 + scikit-learn + PyTorch）
- [ ] 1.4 创建 `paper/figures/` → `../../outputs/figures/` 符号链接
- [ ] 1.5 QA：编译 `xelatex main.tex && bibtex main && xelatex main.tex && xelatex main.tex` 生成 PDF

## 2. 论文正文撰写（5 章并行）

- [ ] 2.1 撰写 `paper/chapters/ch1-intro.tex`——绪论 + 数据集介绍（A同学，~800字）
- [ ] 2.2 撰写 `paper/chapters/ch2-preprocess.tex`——数据预处理（A同学，~800字）
- [ ] 2.3 撰写 `paper/chapters/ch3-ml.tex`——传统ML：DT + RF（B同学，~1200字）
- [ ] 2.4 撰写 `paper/chapters/ch4-dl.tex`——神经网络：MLP/CNN/LSTM（C同学，~1200字）
- [ ] 2.5 撰写 `paper/chapters/ch5-comparison.tex`——对比分析（C同学，~1000字）
- [ ] 2.6 QA：正文总字数 ≥5000（`detex chapters/*.tex | wc -m`）

## 3. 论文收尾

- [ ] 3.1 撰写 `paper/chapters/ch6-conclusion.tex`——总结与展望（三人，~600字）
- [ ] 3.2 补充封面信息（课程名、题目、三人姓名学号）
- [ ] 3.3 最终编译验证 PDF 完整性（封面→目录→6 章→参考文献）

## 4. 项目交付

- [ ] 4.1 完善 `README.md`——添加运行指南（环境配置 + `python scripts/train_m3.py` 等复现步骤）
- [ ] 4.2 更新 `README.md`——项目结构说明、关键结论摘要
- [ ] 4.3 准备答辩 PPT 提纲（`paper/slides_outline.md`）——关键图表选取 + 讲述逻辑
- [ ] 4.4 QA：README 包含完整复现步骤，新用户可从零运行
