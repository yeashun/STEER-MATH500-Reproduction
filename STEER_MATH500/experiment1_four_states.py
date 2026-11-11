"""实验 1：配对已有 small/large 独立回答，分析四状态并导出图表。

统计部分仅依赖 Python 标准库；绘图使用 matplotlib，不加载 GPU 或模型。
状态的第一位是 small、第二位是 large：11 / 01 / 10 / 00。
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics

from analyze_results import read_jsonl, score, single, validate_samples, write_csv

ROOT = Path(__file__).resolve().parent
STATES = ("11", "01", "10", "00")
LABELS = {"11": "都正确", "01": "仅大模型正确", "10": "仅小模型正确", "00": "都错误"}
COLORS = {"11": "#279782", "01": "#3979C3", "10": "#E4A044", "00": "#8993A2"}
PAIR_FIELDS = ("temperature", "top_p", "max_tokens", "max_model_len")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(numerator, denominator):
    return 100 * numerator / denominator if denominator else None


def distribution(rows):
    """所有比例的分母显式定义；没有条件样本时返回 null。"""
    counts = Counter(row["state"] for row in rows)
    total = len(rows)
    if not total:
        raise ValueError("没有可配对的题目")
    n11, n01, n10, n00 = (counts[state] for state in STATES)
    return {
        "num_pairs": total,
        "counts": {state: counts[state] for state in STATES},
        "percentages": {state: percent(counts[state], total) for state in STATES},
        "small_accuracy_pct": percent(n11 + n10, total),
        "large_accuracy_pct": percent(n11 + n01, total),
        "large_minus_small_pp": percent(n01 - n10, total),
        "large_rescue_given_small_wrong_pct": percent(n01, n01 + n00),
        "large_wrong_given_small_correct_pct": percent(n10, n11 + n10),
        "answer_selection_oracle_accuracy_pct": percent(n11 + n01 + n10, total),
    }


def select_ids(dataset, samples, subset_seed):
    ids = sorted(dataset)
    if samples == -1:
        return ids
    if not 1 <= samples <= len(ids):
        raise ValueError(f"samples 必须为 -1 或 1..{len(ids)}")
    # 只依据标准题号抽样，不查看模型得分；相同集合用于全部模型和 seed。
    return sorted(random.Random(subset_seed).sample(ids, samples))


def load_baseline(root, method, seed, dataset, digest):
    directory = root / f"seed_{seed}" / method
    if not (directory / "SUCCESS").is_file():
        raise ValueError(f"未找到成功实验：{directory}")
    config_path = directory / "run_config.json"
    samples_path = directory / "samples.jsonl"
    metrics_path = directory / "metrics.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["method"] != method or config["seed"] != seed or config["samples"] != -1:
        raise ValueError(f"{directory}: 需要相应方法、seed 的完整 500 题基线")
    if config.get("dataset_sha256") != digest:
        raise ValueError(f"{directory}: 数据哈希不匹配")
    if config.get("evaluation", "official_rule_based") != "official_rule_based":
        raise ValueError(f"{directory}: 评分口径不是本项目规则评分")
    if config.get("exit_code", 0) != 0 or config.get("interrupted_by_user", False):
        raise ValueError(f"{directory}: 实验退出状态异常")
    indexed = validate_samples(read_jsonl(samples_path), dataset, method)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    accuracy = 100 * sum(score(row) for row in indexed.values()) / len(dataset)
    if (metrics.get("num_samples") != len(dataset)
            or abs(float(metrics["acc"]) - round(accuracy, 1)) > 1e-7):
        raise ValueError(f"{directory}: 保存的准确率/数量与逐题结果不一致")
    if metrics.get("evaluation", "official_rule_based") != "official_rule_based":
        raise ValueError(f"{directory}: metrics 的评分口径不一致")
    model_key = "draft_model" if method == "small" else "target_model"
    model_name = config[model_key]
    revision = None
    revisions_path = directory / "model_configs.json"
    if revisions_path.is_file():
        metadata = json.loads(revisions_path.read_text(encoding="utf-8"))
        revision = metadata.get(model_name, {}).get("resolved_revision")
    log_path = directory / "run.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    backends = [name for phrase, name in (("Using Flash Attention backend", "FLASH_ATTN"),
                                          ("Using XFormers backend", "XFORMERS")) if phrase in log]
    checks = {
        "method": method, "seed": seed, "directory": str(directory),
        "model": model_name, "resolved_revision": revision,
        "dataset_and_question_content_match": True,
        "saved_accuracy_matches": True, "rule_accuracy_pct": accuracy,
        "rule_timeout_samples": metrics.get("timeout_samples"),
        "attention_backend_requested": config.get("attention_backend_requested", "not_recorded"),
        "attention_backends_observed": backends,
        "budget_semantics": config.get("budget_semantics", "not_recorded"),
        "input_sha256": {path.name: sha256(path) for path in (config_path, samples_path, metrics_path)},
    }
    return {"config": config, "samples": indexed, "checks": checks}


def pair_runs(small, large, dataset, ids, seed):
    for field in PAIR_FIELDS:
        if small["config"][field] != large["config"][field]:
            raise ValueError(f"seed={seed}: small/large 的 {field} 不一致")
    for run in (small, large):
        semantics = run["config"].get("budget_semantics")
        if semantics not in (None, "baseline_generated_tokens"):
            raise ValueError("实验 1 需要独立完整回答的基线，不能混用 STEER 生成预算")
    rows = []
    for idx in ids:
        a, b = small["samples"][idx], large["samples"][idx]
        small_score, large_score = score(a), score(b)
        state = f"{small_score}{large_score}"
        rows.append({
            "seed": seed, "idx": idx, "unique_id": dataset[idx]["unique_id"],
            "level": dataset[idx].get("level", "unknown"),
            "subject": dataset[idx].get("subject", "unknown"),
            "state": state, "state_label": LABELS[state],
            "small_score": small_score, "large_score": large_score,
            "small_pred": single(a, "pred"), "large_pred": single(b, "pred"),
            "small_finish_reason": a.get("finish_reason", "not_recorded"),
            "large_finish_reason": b.get("finish_reason", "not_recorded"),
        })
    return rows


def state_tables(rows, ids, seeds):
    per_seed = [{"seed": seed, **distribution([r for r in rows if r["seed"] == seed])}
                for seed in seeds]
    per_seed_csv = []
    for item in per_seed:
        per_seed_csv.append({"seed": item["seed"], "num_questions": item["num_pairs"],
                             **{f"n_{s}": item["counts"][s] for s in STATES},
                             **{f"pct_{s}": item["percentages"][s] for s in STATES},
                             **{k: v for k, v in item.items()
                                if k not in ("seed", "num_pairs", "counts", "percentages")}})
    stability = []
    by_idx = {idx: [] for idx in ids}
    for row in rows:
        by_idx[row["idx"]].append(row)
    for idx, items in by_idx.items():
        counts = Counter(row["state"] for row in items)
        stability.append({"idx": idx, "unique_id": items[0]["unique_id"],
                          "num_seeds": len(seeds), **{f"n_{s}": counts[s] for s in STATES},
                          "observed_states": ";".join(s for s in STATES if counts[s]),
                          "same_state_all_seeds": len(counts) == 1,
                          "only_large_correct_all_seeds": counts["01"] == len(seeds),
                          "both_wrong_all_seeds": counts["00"] == len(seeds)})
    groups = []
    for field in ("level", "subject"):
        for value in sorted({str(row[field]) for row in rows}):
            group = [row for row in rows if str(row[field]) == value]
            stats = distribution(group)
            groups.append({"group_by": field, "group": value,
                           "num_unique_questions": len({row["idx"] for row in group}),
                           "num_pairs": stats["num_pairs"],
                           **{f"n_{s}": stats["counts"][s] for s in STATES},
                           **{f"pct_{s}": stats["percentages"][s] for s in STATES}})
    return per_seed, per_seed_csv, stability, groups


def plot_settings(language, font_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError as error:
        raise RuntimeError("绘图需要 matplotlib：python -m pip install -r requirements-analysis.txt；"
                           "也可使用 --no-plots 仅生成统计与报告") from error
    font_name = None
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        available = {item.name for item in font_manager.fontManager.ttflist}
        candidates = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                      "WenQuanYi Micro Hei", "PingFang SC", "Arial Unicode MS")
        font_name = next((name for name in candidates if name in available), None)
    if language == "zh" and not font_name:
        raise ValueError("未找到中文字体，请通过 --font-path 指定字体，或使用 --plot-language en")
    chinese = language == "zh" or (language == "auto" and font_name is not None)
    plt.rcParams.update({"font.family": font_name if chinese else "DejaVu Sans",
                         "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "path"})
    return plt, chinese


def plot_results(output, summary, per_seed, groups, settings):
    plt, zh = settings
    from matplotlib.patches import Rectangle, Patch
    labels = LABELS if zh else {"11": "Both correct", "01": "Only large correct",
                               "10": "Only small correct", "00": "Both wrong"}
    pooled = summary["pooled"]
    fig = plt.figure(figsize=(13, 8.8), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.52, wspace=0.42)
    matrix = fig.add_subplot(grid[0, 0])
    matrix.set_title("正确性配对矩阵" if zh else "Paired correctness matrix", loc="left", pad=18)
    for x, y, state in ((0, 1, "11"), (1, 1, "10"), (0, 0, "01"), (1, 0, "00")):
        matrix.add_patch(Rectangle((x, y), 1, 1, facecolor=COLORS[state], edgecolor="white", linewidth=4))
        matrix.text(x + .5, y + .65, state, ha="center", va="center", color="white", fontsize=20, weight="bold")
        matrix.text(x + .5, y + .28, f'{pooled["counts"][state]}  /  {pooled["percentages"][state]:.2f}%',
                    ha="center", va="center", color="white", fontsize=11)
    matrix.set(xlim=(0, 2), ylim=(0, 2), xticks=[.5, 1.5], yticks=[.5, 1.5],
               xticklabels=["正确" if zh else "Correct", "错误" if zh else "Wrong"],
               yticklabels=["错误" if zh else "Wrong", "正确" if zh else "Correct"],
               xlabel="Large LLM", ylabel="Small LLM")
    matrix.set_aspect("equal")
    for spine in matrix.spines.values():
        spine.set_visible(False)
    matrix.tick_params(length=0)
    bars = fig.add_subplot(grid[0, 1])
    heights = [pooled["percentages"][s] for s in STATES]
    errors = [statistics.stdev(item["percentages"][s] for item in per_seed)
              for s in STATES] if len(per_seed) > 1 else [0] * 4
    bars.bar(range(4), heights, color=[COLORS[s] for s in STATES], width=.65,
             yerr=errors if len(per_seed) > 1 else None, capsize=4)
    for i, (height, error) in enumerate(zip(heights, errors)):
        bars.text(i, height + error + 1.5, f"{height:.2f}%", ha="center", weight="bold")
    bars.set(xticks=range(4), xticklabels=[f"{s}\n{labels[s]}" for s in STATES],
             ylabel="配对记录占比（%）" if zh else "Share of paired observations (%)", ylim=(0, 105))
    bars.set_title("四状态分布（误差线：seed 间样本标准差）" if zh else
                   "State distribution (error bars: SD across seeds)", loc="left", pad=18)
    bars.grid(axis="y", alpha=.18)
    bars.set_axisbelow(True)
    stacked = fig.add_subplot(grid[1, :])
    bottom = [0.] * len(per_seed)
    for state in STATES:
        values = [item["percentages"][state] for item in per_seed]
        stacked.barh(range(len(per_seed)), values, left=bottom, color=COLORS[state], height=.6)
        for i, value in enumerate(values):
            if value >= 2:
                annotation = (f'{per_seed[i]["counts"][state]} ({value:.1f}%)' if value >= 7 else
                              str(per_seed[i]["counts"][state]))
                stacked.text(bottom[i] + value / 2, i,
                             annotation, ha="center", va="center",
                             color="white", fontsize=9)
        bottom = [a + b for a, b in zip(bottom, values)]
    stacked.set(yticks=range(len(per_seed)), yticklabels=[f'Seed {item["seed"]}' for item in per_seed],
                xlim=(0, 100), xlabel="每个 seed 内的题目占比（%）" if zh else "Questions within each seed (%)")
    stacked.invert_yaxis()
    stacked.set_title("各 seed 的独立分布" if zh else "Distribution for each seed", loc="left", pad=12)
    handles = [Patch(color=COLORS[s], label=f"{s}  {labels[s]}") for s in STATES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .035))
    title = "实验 1 · 大小模型四状态分布" if zh else "Experiment 1 · Small / Large correctness states"
    subtitle = (f'{summary["num_unique_questions"]} 道固定题目 × {len(per_seed)} 个 seed = '
                f'{pooled["num_pairs"]} 条配对记录（跨 seed 重复不是独立新题目）') if zh else (
                f'{summary["num_unique_questions"]} fixed questions × {len(per_seed)} seeds = '
                f'{pooled["num_pairs"]} paired observations; repeated questions are not independent')
    if summary.get("provenance") == "archived_aggregate_reconstruction":
        subtitle += ("\n由归档三模型汇总重建；尚未读取服务器逐题文件" if zh else
                     "\nReconstructed from archived aggregates; server per-question files not inspected")
    fig.suptitle(title, fontsize=18, weight="bold", y=.995)
    fig.text(.5, .925, subtitle, ha="center", va="center", fontsize=9.5, color="#536071")
    fig.subplots_adjust(top=.85, bottom=.15, left=.09, right=.97)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"four_states.{extension}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if not groups:
        return "zh" if zh else "en"
    # 难度/学科分组使用相同颜色、相同分母定义。
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, field, title in zip(axes, ("level", "subject"),
                               (("按难度分组", "按学科分组") if zh else ("By difficulty level", "By subject"))):
        selected = [item for item in groups if item["group_by"] == field]
        left = [0.] * len(selected)
        for state in STATES:
            values = [item[f"pct_{state}"] for item in selected]
            ax.barh(range(len(selected)), values, left=left, color=COLORS[state], height=.65)
            left = [a + b for a, b in zip(left, values)]
        ax.set(yticks=range(len(selected)),
               yticklabels=[f'{item["group"]} (N={item["num_unique_questions"]})' for item in selected],
               xlim=(0, 100), xlabel="组内配对记录占比（%）" if zh else "Share within group (%)")
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("四状态分组分布" if zh else "Correctness states by question group", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, .08, 1, .93))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"four_states_by_group.{extension}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "zh" if zh else "en"


def report_text(summary, per_seed, stability, plots):
    pooled = summary["pooled"]
    lines = ["# 实验 1：大小模型四状态分布", "",
             f'固定题目数：{summary["num_unique_questions"]}；seed：{summary["seeds"]}；'
             f'配对记录数：{pooled["num_pairs"]}。',
             "每道题由两个模型独立生成完整回答；第一位是 Small，第二位是 Large。",
             "复用原实验保存的规则评分，不重新推理、不重新评分。", "",
             "| 状态 | 含义 | 配对记录数 | 占比 | seed 间占比样本标准差（百分点） |",
             "|---|---|---:|---:|---:|"]
    for state in STATES:
        sd = summary["across_seed_state_percentages"][state]["sample_sd_pp"]
        lines.append(f'| {state} | {LABELS[state]} | {pooled["counts"][state]} | '
                     f'{pooled["percentages"][state]:.2f}% | {sd:.2f} |' if sd is not None else
                     f'| {state} | {LABELS[state]} | {pooled["counts"][state]} | '
                     f'{pooled["percentages"][state]:.2f}% | 单 seed，未定义 |')
    formulas = (
        ("小模型准确率", "small_accuracy_pct", "(11 + 10) / 全部"),
        ("大模型准确率", "large_accuracy_pct", "(11 + 01) / 全部"),
        ("小模型错题中大模型答对比例", "large_rescue_given_small_wrong_pct", "01 / (01 + 00)"),
        ("小模型对题中大模型答错比例", "large_wrong_given_small_correct_pct", "10 / (11 + 10)"),
        ("完整答案事后择优准确率上界", "answer_selection_oracle_accuracy_pct", "(11 + 01 + 10) / 全部"),
    )
    lines += ["", "## 分析指标", ""]
    for label, key, formula in formulas:
        value = pooled[key]
        formatted = f"{value:.2f}%" if value is not None else "未定义（分母为零）"
        lines.append(f"- {label}：**{formatted}**；公式：`{formula}`。")
    lines += [f'- 大模型相对小模型净变化：**{pooled["large_minus_small_pp"]:+.2f} 个百分点**；'
              '公式：`(01 − 10) / 全部 × 100`。', "", "## 各 seed", "",
              "| seed | 题目数 | 11 | 01 | 10 | 00 | Small Acc | Large Acc |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in per_seed:
        counts = item["counts"]
        lines.append(f'| {item["seed"]} | {item["num_pairs"]} | {counts["11"]} | {counts["01"]} | '
                     f'{counts["10"]} | {counts["00"]} | {item["small_accuracy_pct"]:.2f}% | '
                     f'{item["large_accuracy_pct"]:.2f}% |')
    lines += ["", "## 跨 seed 的题目稳定性", "",
              f'- 所选 seed 中始终为 01 的题目：{sum(r["only_large_correct_all_seeds"] for r in stability)}。',
              f'- 所选 seed 中始终为 00 的题目：{sum(r["both_wrong_all_seeds"] for r in stability)}。',
              "- 逐题状态计数见 question_stability.csv；单 seed 只能描述一次观测。", "",
              "## 解释范围", "",
              "- 01 表示这次独立完整回答中大模型答对、小模型答错；不证明在某个中间步骤升级就能救回。",
              "- 00 表示本次两份回答都错，不能推断增加采样、改变提示词或步骤切换仍无法解决。",
              "- 11 表示小模型本次已答对；10 表示大模型也可能失去小模型本次答对的题。",
              "- 事后择优上界假定已知两份回答的正确性，不是可部署路由器的实测准确率。",
              "- 多 seed 是同一组题的重复采样；配对记录不是更多独立题目，误差线不是置信区间。",
              "- 规则评分可能有误判和超时；评分超时、后端和可用模型 revision 见 checks.json。",
              "- 只检查已记录的模型来源；没有重新核验权重字节。", "", "## 图表", ""]
    if plots:
        lines += ["![四状态分布](four_states.png)", "", "![按难度与学科分组](four_states_by_group.png)", "",
                  "同时导出 PDF 与 SVG，便于汇报或论文排版。"]
    else:
        lines += ["本次使用 --no-plots，未生成图；去掉该参数并使用新的输出目录可生成全部图表。"]
    return "\n".join(lines) + "\n"


def analyze(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，请换 --output：{output}")
    if (not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(type(seed) is not int or seed < 0 for seed in args.seeds)):
        raise ValueError("seed 必须非负、非空且不能重复")
    roots = {"small": args.small_root.resolve(), "large": args.large_root.resolve()}
    if any(output == root or root in output.parents for root in roots.values()):
        raise ValueError("分析输出请放在原始结果根目录之外，例如 reports/experiment1_01")
    data_path = ROOT / "data/math500/test.jsonl"
    raw = read_jsonl(data_path)
    dataset = {idx: {"idx": idx, **item} for idx, item in enumerate(raw)}
    digest = sha256(data_path)
    upstream = json.loads((ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))
    if (len(dataset) != 500 or len({r["unique_id"] for r in dataset.values()}) != 500
            or digest != upstream["math500_sha256"]):
        raise ValueError("标准 MATH500 数量、唯一题号或固定哈希不匹配")
    ids = select_ids(dataset, args.samples, args.subset_seed)
    rows, runs = [], []
    for seed in args.seeds:
        print(f"[核对] small / large seed={seed}", flush=True)
        pair = {method: load_baseline(root, method, seed, dataset, digest)
                for method, root in roots.items()}
        rows.extend(pair_runs(pair["small"], pair["large"], dataset, ids, seed))
        runs.extend(pair.values())
    for method in roots:
        selected = [run for run in runs if run["checks"]["method"] == method]
        for field in PAIR_FIELDS + (("draft_model",) if method == "small" else ("target_model",)):
            if len({run["config"][field] for run in selected}) != 1:
                raise ValueError(f"{method}: 跨 seed 的 {field} 不一致")
        revisions = {run["checks"]["resolved_revision"] for run in selected
                     if run["checks"]["resolved_revision"]}
        if len(revisions) > 1:
            raise ValueError(f"{method}: 跨 seed 的模型 revision 不一致")
    per_seed, seed_csv, stability, groups = state_tables(rows, ids, args.seeds)
    summary = {
        "experiment": "small_large_four_states", "state_order": list(STATES),
        "state_definition": "(small_correct, large_correct)", "evaluation": "official_rule_based",
        "num_unique_questions": len(ids), "seeds": args.seeds, "pooled": distribution(rows),
        "across_seed_state_percentages": {
            state: {"mean_pct": statistics.mean(item["percentages"][state] for item in per_seed),
                    "sample_sd_pp": statistics.stdev(item["percentages"][state] for item in per_seed)
                    if len(per_seed) > 1 else None} for state in STATES},
        "per_seed": per_seed,
    }
    # 在写入任何产物前确认绘图依赖/字体，避免缺依赖时留下貌似完整的报告。
    settings = None if args.no_plots else plot_settings(args.plot_language, args.font_path)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_question.csv", rows)
    write_csv(output / "per_seed.csv", seed_csv)
    write_csv(output / "question_stability.csv", stability)
    write_csv(output / "by_level_subject.csv", groups)
    write_csv(output / "selected_questions.csv", [{"idx": idx, "unique_id": dataset[idx]["unique_id"],
                                                  "problem": dataset[idx]["problem"],
                                                  "answer": dataset[idx]["answer"]} for idx in ids])
    with (output / "paired_answers.jsonl").open("w", encoding="utf-8") as stream:
        for seed in args.seeds:
            pair = {r["checks"]["method"]: r for r in runs if r["checks"]["seed"] == seed}
            for row in (r for r in rows if r["seed"] == seed):
                idx = row["idx"]
                record = {**row, "problem": dataset[idx]["problem"], "answer": dataset[idx]["answer"],
                          "small_code": single(pair["small"]["samples"][idx], "code"),
                          "large_code": single(pair["large"]["samples"][idx], "code")}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    language = plot_results(output, summary, per_seed, groups, settings) if settings else None
    checks = {"dataset_sha256": digest, "source_dataset_questions": len(dataset),
              "selection": {"samples_requested": args.samples, "subset_seed": args.subset_seed,
                            "strategy": "all" if args.samples == -1 else "seeded_random_without_replacement",
                            "selected_ids": ids, "selection_does_not_use_model_scores": True},
              "runs": [r["checks"] for r in runs], "plot_language": language,
              "warnings": ["未记录的后端或模型 revision 不能视为已核验。",
                           "本次复用原规则评分；评分超时仍保留在总分母内，不静默删题。"]}
    for name, value in (("summary.json", summary), ("checks.json", checks)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                   encoding="utf-8")
    (output / "report.md").write_text(report_text(summary, per_seed, stability, settings is not None),
                                      encoding="utf-8")
    print("state  count  percentage")
    for state in STATES:
        print(f'{state}  {summary["pooled"]["counts"][state]}  {summary["pooled"]["percentages"][state]:.2f}%')
    print(f"[完成] {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-root", type=Path, default=ROOT / "results/gpu7_small_full_01")
    parser.add_argument("--large-root", type=Path, default=ROOT / "results/gpu7_large_full_01")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--samples", type=int, default=-1, help="-1 为全部 500 题；300 为固定随机子集")
    parser.add_argument("--subset-seed", type=int, default=0, help="抽题 seed，与模型采样 seed 分开")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true", help="仅统计与导出明细，不需要 matplotlib")
    parser.add_argument("--plot-language", choices=["auto", "zh", "en"], default="auto")
    parser.add_argument("--font-path", type=Path, help="可选的中文字体文件路径")
    args = parser.parse_args()
    try:
        analyze(args)
    except (ValueError, KeyError, TypeError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(1, f"实验 1 分析中止：{error}\n")


if __name__ == "__main__":
    main()
