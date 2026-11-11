# 文献解读与复现汇报

[可编辑 PPTX](STEER_literature_and_reproduction.pptx) · [PDF 预览](STEER_literature_and_reproduction.pdf)

共 16 页，按“论文问题 → 方法 → 实现口径 → 复现协议 → 结果 → 后续分析 → 局限与参考文献”重新组织。所有个人结果统一采用共同 seed 0–3，并逐页注明数据或文献来源。

## 逐页提纲

| 页 | 内容 |
|---|---|
| 1 | 文献解读与 MATH500 复现封面 |
| 2 | 论文信息与个人复现范围 |
| 3 | 推理准确率与成本问题 |
| 4 | Query router、RSD、SpecReason、STEER 对比 |
| 5 | 首步重生成与后续步骤路由 |
| 6 | 论文置信度与代码实现差异 |
| 7 | GMM 后验和阈值语义 |
| 8 | 共同 seed 的实验协议 |
| 9 | 论文原表与个人结果分别列示 |
| 10 | 主结果及准确率、FLOPs 差值 |
| 11 | 大小模型四状态计数 |
| 12 | 四状态分布图与 seed 间波动 |
| 13 | 留一 seed 答案一致性代理 |
| 14 | 固定问题的跨 seed 稳定性 |
| 15 | 复现局限与后续实验 |
| 16 | 参考文献与材料入口 |

## 与原稿相比的修订

原稿 `Report_for_steer.pptx` 保留在原工作目录的“文献PPT汇报”中，没有修改。

- 原第 14、15 页的多组结果表无法与本地 AAAI 论文对应，核对版删去这些表，改列论文 Table 1 中 Qwen2.5-Math / MATH500 的原始三行。
- 更正原第 10 页的模型、数据集类型及采样说明；ACPBench 为行动与规划推理，KOR-Bench 覆盖 Cipher / Counterfactual / Logic。原稿的 Gemini / Qwen3 / DeepSeek 组合和贪婪解码设置不作为本次复现设置保留。
- 首步低置信时重生成，后续按前一步置信度选择下一步；更正“每步先小模型再重做”的笼统描述。
- 区分论文高置信后验 γ、代码低置信后验 τ，以及最大 logit 与 R_eu 的实现差异。
- 主对照统一 seed 0–3；基线第五个 seed 和 XFormers 的 STEER seed 4 留在补充元数据中。
- 移除没有显著性检验依据的“显著提高”表述；准确率差值用百分点表示，FLOPs 不解释成延迟或能耗。
- 加入正式四状态分析、答案一致性代理、跨 seed 稳定性与可复核材料入口。
- 汇报日期标为材料整理日期 2026-11-11；不沿用原稿占位日期。

## 再生成

布局和文字在 [content.json](content.json) 中，PPT 中的文字与表格均可编辑，四状态图作为图片嵌入。

```bash
python -m pip install -r scripts/requirements-slides.txt
python scripts/build_slides.py
```

以上命令在仓库根目录执行。默认使用 Windows 的微软雅黑字体；其他系统通过 `--font` 和 `--bold-font` 指定支持中文的 TrueType 字体。PDF 嵌入字体子集。生成器检查文字高度，并用同一份布局生成 PPTX 和 PDF；本次逐页图像检查基于 PDF，未能在本机 PowerPoint 中完成打开与播放检查。
