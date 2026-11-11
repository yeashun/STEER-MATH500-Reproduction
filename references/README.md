# 论文与相关工作

| 文献 | 在本项目中的作用 | 原始来源 |
|---|---|---|
| Lee et al., 2026. Confidence-Guided Stepwise Model Routing for Cost-Efficient Reasoning | 核心论文，AAAI 40(37): 31483–31491 | [出版页](https://ojs.aaai.org/index.php/AAAI/article/view/40413) · [DOI](https://doi.org/10.1609/aaai.v40i37.40413) · [arXiv](https://arxiv.org/abs/2511.06190) |
| STEER official implementation | 算法、vLLM 补丁、数据与原始评测器来源 | [固定版本](https://github.com/helmsman12/STEER/tree/04f5ad59bafa394b449f3e74e1b0529667a02cc7) |
| Yang et al., 2024. Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement | 数学模型系列 | [arXiv:2409.12122](https://arxiv.org/abs/2409.12122) |
| Liao et al., 2025. Reward-Guided Speculative Decoding for Efficient LLM Reasoning | 依赖 PRM 的步骤级方法对照；本项目未复跑 | [arXiv:2501.19324](https://arxiv.org/abs/2501.19324) |
| Pan et al., 2025. SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning | 使用大模型评估草稿步骤的相关工作；本项目未复跑 | [arXiv:2504.07891](https://arxiv.org/abs/2504.07891) |

使用 [references.bib](references.bib) 引用。正式论文与 arXiv 版本应区分；复现依据本地 AAAI PDF 与上述固定源码，不假设当前上游 HEAD 的行为完全一致。MATH500 文件来自固定上游版本，SHA256 见复现指南。

汇报中的论文结果与个人复现结果分别标注，避免混引。未将原始论文 PDF 重复放入仓库。
