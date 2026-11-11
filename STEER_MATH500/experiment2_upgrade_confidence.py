"""实验 2：仅用已有 small/large 回答分析 01/00，并统计跨 seed 状态。

旧基线没有保存概率置信度。本实验使用不读取标准答案或 score 的
leave-one-seed-out 答案一致率作为离线代理，不将它冒充 token 置信度。
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np

import experiment1_four_states as exp1
from analyze_results import read_jsonl, single

ROOT = Path(__file__).resolve().parent
SEEDS = (0, 1, 2, 3)
STATES = exp1.STATES
CONFIDENCE_FIELDS = {"logits", "top_logits_list", "logprobs", "token_logprobs",
                     "cumulative_logprob", "sequence_logprob", "mean_logprob",
                     "confidence", "step_reliabilities"}


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"不能导出空表：{path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_prediction(value):
    # 只删除空白，不使用标准答案、score 或符号评分器。
    return "" if value is None else "".join(str(value).split())


def answer_agreement(small_by_seed, ids):
    """输入仅为小模型的 pred；自身 seed 不计入一致票数。"""
    predictions = {(seed, idx): normalized_prediction(single(small_by_seed[seed][idx], "pred"))
                   for seed in SEEDS for idx in ids}
    result = {}
    for (seed, idx), current in predictions.items():
        matches = sum(bool(current) and current == predictions[other, idx]
                      for other in SEEDS if other != seed)
        result[seed, idx] = {
            "normalized_prediction": current,
            "other_seed_matches": matches if current else None,
            "answer_agreement_proxy": matches / 3 if current else None,
            "proxy_status": "available" if current else "missing_empty_prediction",
        }
    return result


def auc(labels, scores):
    """较高 score 预测正类；相同分数按半胜计算。"""
    if len(labels) != len(scores) or any(y not in (0, 1) for y in labels):
        raise ValueError("AUROC 输入长度或标签非法")
    if any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("AUROC 不接受非有限分数")
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return None
    grouped = {}
    for y, value in zip(labels, scores):
        grouped.setdefault(value, [0, 0])[y] += 1
    concordant = below_negatives = 0
    for value in sorted(grouped):
        negative, positive = grouped[value]
        concordant += positive * (below_negatives + .5 * negative)
        below_negatives += negative
    return concordant / (positives * negatives)


def cluster_bootstrap(rows, question_ids, repetitions, seed):
    """按题目抽样，保留题内全部 seed；不把 513 条记录当作独立样本。"""
    if repetitions < 100:
        raise ValueError("bootstrap 次数至少为 100")
    ids = list(question_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("聚类题号不能重复")
    positions = {idx: position for position, idx in enumerate(ids)}
    # 最后一维：00 负类、01 正类。四个分数档位固定为 0/3,1/3,2/3,3/3。
    counts = np.zeros((len(ids), 4, 2), dtype=np.int64)
    for row in rows:
        if row["answer_agreement_proxy"] is not None:
            counts[positions[row["idx"]], row["other_seed_matches"], int(row["state"] == "01")] += 1
    rng = np.random.default_rng(seed)
    # 每一行的权重均为完整题目被抽中的次数，共重采样 500 题。
    weights = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=repetitions)
    negative = weights @ counts[:, :, 0]
    positive = weights @ counts[:, :, 1]
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    valid = denominator > 0
    numerator = (positive * (np.cumsum(negative, axis=1) - .5 * negative)).sum(axis=1)
    values = numerator[valid] / denominator[valid]
    if not len(values):
        raise ValueError("所有 bootstrap 样本都缺少一个类别")
    low, high = np.quantile(values, [.025, .975])
    return {"method": "question_cluster_percentile_bootstrap",
            "sampling_unit": "question; all four seed observations kept together",
            "num_source_questions": len(ids), "repetitions": repetitions,
            "valid_repetitions": int(valid.sum()), "seed": seed,
            "higher_agreement_auc_ci95": [float(low), float(high)],
            "lower_agreement_auc_ci95": [float(1 - high), float(1 - low)]}


def summarize_proxy(selected, ids, repetitions, bootstrap_seed):
    covered = [r for r in selected if r["answer_agreement_proxy"] is not None]
    groups = {state: [r for r in covered if r["state"] == state] for state in ("01", "00")}
    if not all(groups.values()):
        raise ValueError("01 和 00 均需要有可用代理分数")
    histograms = {state: [sum(r["other_seed_matches"] == value for r in group) for value in range(4)]
                  for state, group in groups.items()}
    proportions = {state: [n / len(groups[state]) for n in histograms[state]] for state in groups}
    labels = [int(r["state"] == "01") for r in covered]
    scores = [r["answer_agreement_proxy"] for r in covered]
    pooled_auc = auc(labels, scores)
    by_seed = []
    for seed in SEEDS:
        group = [r for r in covered if r["seed"] == seed]
        value = auc([int(r["state"] == "01") for r in group], [r["answer_agreement_proxy"] for r in group])
        by_seed.append({"seed": seed, "n_01": sum(r["state"] == "01" for r in group),
                        "n_00": sum(r["state"] == "00" for r in group),
                        "auc_higher_agreement_predicts_01": value,
                        "auc_lower_agreement_predicts_01": None if value is None else 1 - value})
    return {"positive_class": "01", "negative_class": "00",
            "proxy": "leave_one_seed_out_normalized_answer_agreement",
            "proxy_direction": "higher score means more agreement, not an outcome-selected sign",
            "is_token_or_sequence_probability": False,
            "selected_records": len(selected), "selected_unique_questions": len({r["idx"] for r in selected}),
            "available_records": len(covered), "missing_records": len(selected) - len(covered),
            "class_counts_all": {s: sum(r["state"] == s for r in selected) for s in groups},
            "class_counts_available": {s: len(groups[s]) for s in groups},
            "agreement_match_counts": histograms, "within_class_proportions": proportions,
            "auc_higher_agreement_predicts_01": pooled_auc,
            "auc_lower_agreement_predicts_01": 1 - pooled_auc,
            "discrete_distribution_overlap": sum(min(a, b) for a, b in zip(proportions["01"], proportions["00"])),
            "bootstrap": cluster_bootstrap(selected, ids, repetitions, bootstrap_seed), "by_seed": by_seed,
            "h1_status": "True probability confidence unavailable; proxy experiment only, not a definitive H1 test"}


def question_patterns(pairs, dataset):
    indexed = {(row["seed"], row["idx"]): row for row in pairs}
    rows = []
    for idx in sorted(dataset):
        states = [indexed[seed, idx]["state"] for seed in SEEDS]
        counts = Counter(states)
        errors = counts["01"] + counts["00"]
        rows.append({"idx": idx, "unique_id": dataset[idx]["unique_id"],
                     "problem": dataset[idx]["problem"], "level": dataset[idx].get("level"),
                     "subject": dataset[idx].get("subject"),
                     **{f"state_seed_{seed}": state for seed, state in zip(SEEDS, states)},
                     **{f"n_{state}": counts[state] for state in STATES},
                     "num_distinct_states": len(counts), "pattern": "|".join(states),
                     "category": "stable_" + states[0] if len(counts) == 1 else "mixed",
                     "large_rescue_rate_given_small_wrong": counts["01"] / errors if errors else None,
                     "four_times_00": counts["00"] == 4,
                     "one_00_three_01": counts["00"] == 1 and counts["01"] == 3})
    categories = Counter(row["category"] for row in rows)
    n_states = Counter(row["num_distinct_states"] for row in rows)
    summary = {"num_questions": len(rows), "num_seeds": 4,
               "categories": {key: categories[key] for key in ["stable_" + s for s in STATES] + ["mixed"]},
               "num_distinct_states_counts": {str(k): n_states[k] for k in range(1, 5)},
               "four_times_00": sum(r["four_times_00"] for r in rows),
               "one_00_three_01": sum(r["one_00_three_01"] for r in rows)}
    return rows, summary


def plot_results(output, proxy, patterns, stability, language, font_path):
    plt, zh = exp1.plot_settings(language, font_path)
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    colors = {state: exp1.COLORS[state] for state in ("01", "00")}
    labels = {"01": "01 有效升级", "00": "00 本次共同失败"} if zh else {
        "01": "01: only large correct", "00": "00: both wrong"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    grid = np.arange(4)
    for offset, state in ((-.18, "01"), (.18, "00")):
        proportions = proxy["within_class_proportions"][state]
        axes[0].bar(grid + offset, np.array(proportions) * 100, width=.34, color=colors[state],
                    label=f'{labels[state]} (n={proxy["class_counts_available"][state]})')
        for x, value in zip(grid + offset, proportions):
            axes[0].text(x, value * 100 + 1.2, f"{value * 100:.1f}%", ha="center", fontsize=8)
        cdf = np.cumsum(proportions)
        axes[1].step([-.05, 0, 1 / 3, 2 / 3, 1, 1.05], [0, *cdf, 1], where="post",
                     color=colors[state], label=labels[state], linewidth=2)
    axes[0].set(xticks=grid, xticklabels=["0/3", "1/3", "2/3", "3/3"], ylim=(0, 75),
                xlabel="与另外 3 个 seed 的答案一致率" if zh else "Agreement with other 3 seeds",
                ylabel="各类内部占比（%）" if zh else "Percentage within class")
    axes[0].set_title("代理分布（非概率置信度）" if zh else "Proxy distribution (not probability)", loc="left")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(xlim=(-.05, 1.05), ylim=(0, 1.05),
                xlabel="答案一致率代理" if zh else "Answer-agreement proxy", ylabel="ECDF")
    axes[1].set_title("累积分布重叠" if zh else "Empirical cumulative distributions", loc="left")
    axes[1].text(.98, .08, f'Overlap = {proxy["discrete_distribution_overlap"]:.3f}',
                 transform=axes[1].transAxes, ha="right")
    axes[1].legend(frameon=False, fontsize=8)
    negative = np.array(proxy["agreement_match_counts"]["00"])[::-1]
    positive = np.array(proxy["agreement_match_counts"]["01"])[::-1]
    fpr = np.r_[0, np.cumsum(negative) / negative.sum()]
    tpr = np.r_[0, np.cumsum(positive) / positive.sum()]
    axes[2].plot(fpr, tpr, color=colors["01"], marker="o", label="Higher agreement predicts 01")
    axes[2].plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1)
    low, high = proxy["bootstrap"]["higher_agreement_auc_ci95"]
    axes[2].set(xlim=(0, 1), ylim=(0, 1.05), xlabel="FPR (00)", ylabel="TPR (01)")
    axes[2].set_title(f'ROC: AUROC = {proxy["auc_higher_agreement_predicts_01"]:.3f}', loc="left")
    axes[2].text(.98, .08, f"Question-cluster 95% CI\n[{low:.3f}, {high:.3f}]",
                 transform=axes[2].transAxes, ha="right", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("实验 2 · 01 与 00 的小模型答案一致性代理" if zh else
                 "Experiment 2 · Small-model answer-agreement proxy: 01 vs 00", fontsize=16, weight="bold", y=.99)
    subtitle = (f'{proxy["selected_records"]} 条记录 / {proxy["selected_unique_questions"]} 道题；'
                "仅复用已保存答案，未保存 logits 或序列概率") if zh else (
                f'{proxy["selected_records"]} observations / {proxy["selected_unique_questions"]} questions; '
                "saved answers only, no logits or sequence probabilities")
    fig.text(.5, .88, subtitle, ha="center", fontsize=9.5, color="#536071")
    fig.tight_layout(rect=(0, 0, 1, .82))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"confidence_proxy_01_vs_00.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={"width_ratios": [1, 1.1], "hspace": .48, "wspace": .38})
    ordered = sorted(patterns, key=lambda r: (-r["n_00"], -r["n_01"], -r["n_10"], r["idx"]))
    matrix = np.array([[STATES.index(row[f"state_seed_{seed}"]) for seed in SEEDS] for row in ordered])
    axes[0, 0].imshow(matrix, aspect="auto", interpolation="nearest",
                       cmap=ListedColormap([exp1.COLORS[s] for s in STATES]), vmin=-.5, vmax=3.5)
    axes[0, 0].set(xticks=range(4), xticklabels=[f"Seed {s}" for s in SEEDS],
                   ylabel="固定题目（按状态次数排序）" if zh else "Fixed questions, sorted by state counts")
    axes[0, 0].set_title("500 题 × 4 seed 状态图" if zh else "500 questions × 4 seeds", loc="left")
    category_keys = ["stable_" + s for s in STATES] + ["mixed"]
    values = [stability["categories"][key] for key in category_keys]
    axes[0, 1].bar(range(5), values, color=[exp1.COLORS[s] for s in STATES] + ["#9C78B5"])
    axes[0, 1].set(xticks=range(5), xticklabels=[("始终 " if zh else "Always ") + s for s in STATES] + ["有变化" if zh else "Mixed"],
                   ylabel="题目数" if zh else "Number of questions", ylim=(0, max(values) * 1.2))
    axes[0, 1].set_title("四次状态是否稳定" if zh else "Stable vs changing states", loc="left")
    for i, value in enumerate(values):
        axes[0, 1].text(i, value + 5, str(value), ha="center")
    joint = np.zeros((5, 5))
    for row in patterns:
        joint[row["n_01"], row["n_00"]] += 1
    masked = np.ma.array(joint, mask=np.fromfunction(lambda y, x: x + y > 4, (5, 5)))
    axes[1, 0].imshow(masked, origin="lower", cmap="Blues", aspect="auto")
    for y in range(5):
        for x in range(5 - y):
            axes[1, 0].text(x, y, str(int(joint[y, x])), ha="center", va="center",
                            color="white" if joint[y, x] > joint.max() / 2 else "black")
    axes[1, 0].set(xticks=range(5), yticks=range(5),
                   xlabel="每题 00 出现次数" if zh else "00 occurrences per question",
                   ylabel="每题 01 出现次数" if zh else "01 occurrences per question")
    axes[1, 0].set_title("00/01 次数联合分布（格内为题目数）" if zh else "Joint 00 / 01 counts (cell = questions)", loc="left")
    values = [stability["num_distinct_states_counts"][str(i)] for i in range(1, 5)]
    axes[1, 1].bar(range(1, 5), values, color="#527BA6")
    axes[1, 1].set(xticks=range(1, 5), xlabel="同一道题出现的不同状态数" if zh else "Number of distinct states per question",
                   ylabel="题目数" if zh else "Number of questions", ylim=(0, max(values) * 1.15))
    axes[1, 1].set_title("跨 seed 的状态变化" if zh else "Variation across seeds", loc="left")
    for i, value in enumerate(values, 1):
        axes[1, 1].text(i, value + 5, str(value), ha="center")
    handles = [Patch(color=exp1.COLORS[s], label=s) for s in STATES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("固定 500 题的跨 seed 状态稳定性" if zh else "State stability of the fixed 500 questions", fontsize=17, weight="bold", y=.99)
    fig.text(.5, .94, "seed 是采样重复，不是时间轴；稳定仅指已观察的四次回答" if zh else
             "Seeds are sampling replicates, not time; stability refers only to these four observations", ha="center", fontsize=10)
    fig.subplots_adjust(top=.87, bottom=.10)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"question_state_stability.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def report_text(proxy, stability, field_audit, plots):
    ci = proxy["bootstrap"]["higher_agreement_auc_ci95"]
    low_ci = proxy["bootstrap"]["lower_agreement_auc_ci95"]
    lines = ["# 实验 2：01/00 区分能力与跨 seed 状态稳定性", "",
             "## 数据与置信度可用性", "",
             "来源是实验 1 从服务器只读下载的 small/large seed 0–3 原始逐题文件；没有重新生成、重评分或训练路由器。",
             f'仅对 01/00 计算 AUROC：01={proxy["class_counts_all"]["01"]}，00={proxy["class_counts_all"]["00"]}，'
             f'合计 {proxy["selected_records"]} 条，涉及 {proxy["selected_unique_questions"]} 道题。',
             f'小模型记录数 {field_audit["num_small_records"]}；未保存 token logits、token logprob 或序列概率。'
             '基线生成入口也没有请求 logprobs，不能从现有文本反推出真正的概率置信度。', "",
             "因此本次是**答案一致性代理实验**，不能直接作为论文 logit 置信度的 H1 检验。", "",
             "## 预先定义的代理", "",
             "对题目 q、seed s，把小模型提取答案删除空白后，与另外三个 seed 的小模型答案比较：",
             "`proxy(q,s) = 与当前答案相同的其他 seed 数 / 3`。",
             "自身 seed 不投票；特征不读取标准答案、Small/Large score、大模型答案或四状态标签。",
             "仅以字符串相同判定一致，不进行数学等价化；等价但写法不同会被记为不一致。空答案记缺失，不能把两个空答案当作高一致性。",
             f'可用 {proxy["available_records"]} 条，缺失 {proxy["missing_records"]} 条。',
             "该代理需要同题四次小模型生成，是事后多采样分析，不是单次生成即可部署的路由特征；没有使用模型生成的自评信心。", "",
             "## 分布与 AUROC", "",
             "正类固定为 01；默认方向为一致率越高越预测 01，同时报告反方向，不能根据标签选择方向后只报告较好的一边。", "",
             "| 与另外三个 seed 相同的票数 | 01 数量 | 01 类内占比 | 00 数量 | 00 类内占比 |",
             "|---|---:|---:|---:|---:|"]
    for value in range(4):
        lines.append(f'| {value}/3 | {proxy["agreement_match_counts"]["01"][value]} | '
                     f'{proxy["within_class_proportions"]["01"][value] * 100:.2f}% | '
                     f'{proxy["agreement_match_counts"]["00"][value]} | '
                     f'{proxy["within_class_proportions"]["00"][value] * 100:.2f}% |')
    lines += ["", f'- 高一致率预测 01：AUROC **{proxy["auc_higher_agreement_predicts_01"]:.4f}**；'
              f'按题目聚类 bootstrap 95% 区间 **[{ci[0]:.4f}, {ci[1]:.4f}]**。',
              f'- 低一致率预测 01：AUROC **{proxy["auc_lower_agreement_predicts_01"]:.4f}**；'
              f'区间 **[{low_ci[0]:.4f}, {low_ci[1]:.4f}]**。',
              f'- 离散分布重叠系数：**{proxy["discrete_distribution_overlap"]:.4f}**；'
              '定义为 `sum_v min(P(proxy=v|01), P(proxy=v|00))`，范围 0–1，越大越重叠。',
              f'- bootstrap 重采样 {proxy["bootstrap"]["repetitions"]} 次，随机种子 '
              f'{proxy["bootstrap"]["seed"]}；每次重采样 500 道完整题目，再取 01/00。',
              "- 同题四个 seed 一起重采样，区间描述题目重采样下的稳定性，不把 513 条记录当作独立题目；没有额外估计新 seed 的变异。", "",
              "| seed | 01 | 00 | 高一致率预测 01 AUROC | 低一致率预测 01 AUROC |",
              "|---|---:|---:|---:|---:|"]
    for row in proxy["by_seed"]:
        high = row["auc_higher_agreement_predicts_01"]
        low = row["auc_lower_agreement_predicts_01"]
        lines.append(f'| {row["seed"]} | {row["n_01"]} | {row["n_00"]} | '
                     f'{high:.4f} | {low:.4f} |' if high is not None else
                     f'| {row["seed"]} | {row["n_01"]} | {row["n_00"]} | 未定义 | 未定义 |')
    lines += ["", "## 跨 seed 的状态稳定性", "",
              "每道题完整保留 seed 0、1、2、3 的四状态；seed 是采样重复，不能解释为时间上的能力演化。", "",
              "| 题目类别 | 题目数 |", "|---|---:|"]
    for state in STATES:
        lines.append(f'| 四次始终 {state} | {stability["categories"]["stable_" + state]} |')
    lines += [f'| 状态有变化 | {stability["categories"]["mixed"]} |',
              f'| 合计 | {stability["num_questions"]} |', "",
              f'四次均为 00 的题目 **{stability["four_times_00"]}** 道；'
              f'一次 00、三次 01 的题目 **{stability["one_00_three_01"]}** 道。',
              "具体题号、问题、四次状态与次数见 question_state_patterns.csv；两类例子分别导出 stable_00_questions.csv 和 one00_three01_questions.csv。",
              "这些是有限四次观察下的稳定性，不构成不可解证明；同一题的改变也可能来自评分或采样差异。", "",
              "## 对 H1 的解释", ""]
    if ci[0] <= .5 <= ci[1]:
        lines += ["当前代理的区分证据不足：高一致率方向的 AUROC 区间覆盖 0.5。结合分布重叠，应报告“该离线答案一致率代理未显示可靠的有效升级识别能力”。"]
    else:
        lines += ["当前代理的 AUROC 区间没有覆盖 0.5，应结合预先定义的方向、效应大小与跨 seed 波动描述其区分能力。"]
    lines += ["这既不能证明真正的 logits/序列概率置信度无效，也不能证明复杂路由器有必要。H1 对真正概率置信度仍待验证。",
              "若要直接检验 H1，下一步应对已保存的小模型回答补算 teacher-forced token/序列概率并单独记录口径，或在新生成实验中保存这些值；本次没有执行模型评分。",
              "后续比较问题特征或升级收益估计时，应按题号划分训练/测试集，同题不同 seed 不能跨集合泄漏。", ""]
    if plots:
        lines += ["![01/00 答案一致性代理](confidence_proxy_01_vs_00.png)", "",
                  "![固定题目跨 seed 状态](question_state_stability.png)", ""]
    return "\n".join(lines)


def analyze(args):
    output = args.output.resolve()
    source = args.source.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请换 --output：{output}")
    if source == output or source in output.parents:
        raise ValueError("输出必须放在原始核验目录之外")
    provenance_path = source / "remote_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for item in provenance["downloaded_files"]:
        if exp1.sha256(source / item["local_path"]) != item["sha256"]:
            raise ValueError(f'输入副本 SHA256 不匹配：{item["local_path"]}')
    dataset_path = ROOT / "data/math500/test.jsonl"
    dataset = {idx: {"idx": idx, **row} for idx, row in enumerate(read_jsonl(dataset_path))}
    digest = exp1.sha256(dataset_path)
    if digest != provenance["dataset_matches_local_frozen_sha256"] or len(dataset) != 500:
        raise ValueError("固定 MATH500 数据不匹配")
    ids = sorted(dataset)
    all_pairs, small_by_seed, runs = [], {}, []
    for seed in SEEDS:
        pair = {method: exp1.load_baseline(source / "input_sources" / method, method, seed, dataset, digest)
                for method in ("small", "large")}
        small_by_seed[seed] = pair["small"]["samples"]
        all_pairs.extend(exp1.pair_runs(pair["small"], pair["large"], dataset, ids, seed))
        runs.extend(pair.values())
    features = answer_agreement(small_by_seed, ids)
    selected = [{**row, **features[row["seed"], row["idx"]]} for row in all_pairs if row["state"] in ("01", "00")]
    fields = Counter(key for samples in small_by_seed.values() for sample in samples.values() for key in sample)
    field_audit = {"num_small_records": sum(len(samples) for samples in small_by_seed.values()),
                   "top_level_field_coverage": dict(fields),
                   "confidence_fields_present": sorted(CONFIDENCE_FIELDS.intersection(fields)),
                   "baseline_source_sha256": exp1.sha256(ROOT / "baseline.py")}
    if field_audit["confidence_fields_present"]:
        raise ValueError("发现置信度字段，应先明确字段语义；不要自动替换为答案一致性代理")
    proxy = summarize_proxy(selected, ids, args.bootstrap, args.bootstrap_seed)
    patterns, stability = question_patterns(all_pairs, dataset)
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output / "selected_01_00.csv", selected)
    save_csv(output / "auc_by_seed.csv", proxy["by_seed"])
    save_csv(output / "question_state_patterns.csv", patterns)
    for filename, flag in (("stable_00_questions.csv", "four_times_00"),
                           ("one00_three01_questions.csv", "one_00_three_01")):
        matching = [row for row in patterns if row[flag]]
        if matching:
            save_csv(output / filename, matching)
    compositions = Counter(tuple(row[f"n_{s}"] for s in STATES) for row in patterns)
    save_csv(output / "state_composition_counts.csv", [{**{f"n_{s}": n for s, n in zip(STATES, counts)}, "num_questions": n}
                                                       for counts, n in sorted(compositions.items())])
    ordered_patterns = Counter(row["pattern"] for row in patterns)
    save_csv(output / "ordered_state_pattern_counts.csv", [{"state_seed_0_to_3": pattern, "num_questions": n}
                                                           for pattern, n in ordered_patterns.most_common()])
    plots = not args.no_plots
    if plots:
        plot_results(output, proxy, patterns, stability, args.plot_language, args.font_path)
    summary = {"confidence_proxy_analysis": proxy, "question_state_stability": stability}
    checks = {"source_directory": str(source), "source_provenance_sha256": exp1.sha256(provenance_path),
              "downloaded_source_hashes_verified": len(provenance["downloaded_files"]),
              "field_audit": field_audit, "dataset_sha256": digest, "seeds": list(SEEDS),
              "runs": [run["checks"] for run in runs],
              "feature_uses_ground_truth_or_scores": False, "empty_answers_are_missing": True,
              "no_model_generation_or_scoring": True, "numpy_version": np.__version__}
    for name, value in (("summary.json", summary), ("checks.json", checks)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(report_text(proxy, stability, field_audit, plots), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[完成] {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "reports/experiment1_remote_check_20261023_01")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--plot-language", choices=["auto", "zh", "en"], default="auto")
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    try:
        analyze(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, RuntimeError) as error:
        parser.exit(1, f"实验 2 中止：{error}\n")


if __name__ == "__main__":
    main()
