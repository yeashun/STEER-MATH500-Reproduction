# MATH500 主实验：共同 seed 0–3

三种方法均使用同一组 500 题、四个 seed；准确率为规则评分，标准差按 seed 计算。

| 方法 | 准确率均值 ± 样本标准差（%） | FLOPs/题均值 ± 样本标准差（T） |
|---|---:|---:|
| small | 74.35 ± 0.68 | 1.679000 ± 0.035761 |
| large | 81.50 ± 1.48 | 10.799033 ± 0.324231 |
| STEER | 78.15 ± 1.20 | 4.679137 ± 0.557911 |

STEER 相对 small 提高 3.80 个百分点，相对 large 下降 3.35 个百分点；相对 large 的均值 FLOPs 节省 56.67%。没有做显著性检验。

成本按 `2 × 1.5B × (small_used + small_discarded) + 2 × 7B × large_generated` 计算，再按题目求均值。T = 10¹²；不代表延迟或能耗测量。

## 来源与核验

2026-11-11 只读下载服务器 STEER seed 0–3 的原始逐题文件与现有审计报告；small/large 使用 2026-10-23 已核验的本地原始副本。本次从 12 组逐题文件重新计算准确率和 token 成本，与服务器已有审计均值一致。来源 SHA256 见 [input-files.json](../../provenance/input-files.json)。

服务器历史审计报告称同一规则评测器重新评分变化为 0；本次没有重新评分、调用 xVerify 或运行 GPU 推理。原始完整回答与日志保留在原工作目录和服务器；发布版仅保留重算所需的 [score_token_counts.csv](score_token_counts.csv)、逐 seed 统计及既有审计结果。

## 比较限制

- STEER 的 token 预算包含 prompt，基线限制生成 token，不能称为严格等预算实验。
- 历史审计有 4 组运行未从初始化日志确认 FlashAttention；缺少记录不等于使用了其他后端，见 checks.json。
- 7B 权重字节未重新核验；部分本地路径的 resolved revision 缺失。
- STEER 的逐题停止原因不完整；同题跨 seed 记录并非独立新题目。
- 基线 seed 4 与 XFormers 的 STEER seed 4 是补充实验，均不纳入此主表。
