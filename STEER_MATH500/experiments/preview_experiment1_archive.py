"""从 README 归档的三模型配对计数重建实验 1 预览；不生成虚构逐题数据。

在 STEER_MATH500 下运行：
python experiments/preview_experiment1_archive.py --output reports/experiment1_archive_preview_01
正式逐题核验入口是 experiment1_four_states.py。
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment1_four_states import STATES, LABELS, distribution, plot_results, plot_settings, sha256
from analyze_results import write_csv

# 来源：README 的 small/large seed 表、STEER seed 表、2026-10-09 逐题核对表。
# STEER 正确题数 = 对应准确率 × 500，均为精确整数。
ARCHIVE = [
    dict(seed=0, total=500, small_correct=375, large_correct=414, steer_correct=389,
         rescues_small=38, loses_small=24, loses_large=42, rescues_large=17,
         all_wrong=60, all_correct=340),
    dict(seed=1, total=500, small_correct=372, large_correct=397, steer_correct=396,
         rescues_small=49, loses_small=25, loses_large=34, rescues_large=33,
         all_wrong=59, all_correct=330),
    dict(seed=2, total=500, small_correct=367, large_correct=411, steer_correct=383,
         rescues_small=38, loses_small=22, loses_large=48, rescues_large=20,
         all_wrong=61, all_correct=336),
    dict(seed=3, total=500, small_correct=373, large_correct=408, steer_correct=395,
         rescues_small=43, loses_small=21, loses_large=33, rescues_large=20,
         all_wrong=62, all_correct=342),
]


def reconstruct(record):
    """恢复三变量二元列联表，并核对全部边际计数。"""
    r = record
    n110 = r["loses_small"] + r["loses_large"] - (r["total"] - r["all_wrong"] - r["steer_correct"])
    n001 = r["rescues_small"] + r["rescues_large"] - (r["steer_correct"] - r["all_correct"])
    # 此处三位的顺序为 Small, Large, STEER。
    cells = {"111": r["all_correct"], "000": r["all_wrong"], "110": n110, "001": n001,
             "100": r["loses_small"] - n110, "010": r["loses_large"] - n110,
             "011": r["rescues_small"] - n001, "101": r["rescues_large"] - n001}
    if any(type(value) is not int or value < 0 for value in cells.values()):
        raise ValueError("归档计数不能构成非负整数列联表")
    if sum(cells.values()) != r["total"]:
        raise ValueError("归档总题数不一致")
    for position, key in enumerate(("small_correct", "large_correct", "steer_correct")):
        if sum(n for state, n in cells.items() if state[position] == "1") != r[key]:
            raise ValueError(f"归档 {key} 边际计数不一致")
    return {state: sum(n for triple, n in cells.items() if triple[:2] == state) for state in STATES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    per_seed = []
    pooled_rows = []
    for record in ARCHIVE:
        counts = reconstruct(record)
        # 只展开状态频次计算比例，不构造题号、答案或不存在的逐题对应。
        states = [{"state": state} for state in STATES for _ in range(counts[state])]
        per_seed.append({"seed": record["seed"], **distribution(states)})
        pooled_rows.extend(states)
    summary = {
        "experiment": "small_large_four_states", "provenance": "archived_aggregate_reconstruction",
        "num_unique_questions": 500, "seeds": [r["seed"] for r in ARCHIVE],
        "state_order": list(STATES), "state_definition": "(small_correct, large_correct)",
        "evaluation": "official_rule_based", "pooled": distribution(pooled_rows), "per_seed": per_seed,
        "across_seed_state_percentages": {
            s: {"mean_pct": statistics.mean(r["percentages"][s] for r in per_seed),
                "sample_sd_pp": statistics.stdev(r["percentages"][s] for r in per_seed)} for s in STATES},
    }
    settings = plot_settings("auto", None)
    output.mkdir(parents=True, exist_ok=False)
    plot_results(output, summary, per_seed, [], settings)
    source = {"source": "README.md：服务器逐题核对报告（2026-10-09）及各模型 seed 表",
              "readme_sha256": sha256(ROOT / "README.md"), "archive_records": ARCHIVE,
              "server_raw_files_inspected_locally": False,
              "limitations": "只能重建计数，不能恢复题号、回答、分组或跨 seed 题目稳定性。"}
    for name, content in (("summary.json", summary), ("archive_source.json", source)):
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "per_seed.csv", [{"seed": r["seed"],
                                       **{f"n_{s}": r["counts"][s] for s in STATES},
                                       **{f"pct_{s}": r["percentages"][s] for s in STATES}} for r in per_seed])
    pooled = summary["pooled"]
    lines = ["# 实验 1：归档汇总重建预览", "",
             "来源：README 保存的服务器汇总。没有读取服务器原始逐题文件。",
             "500 道固定题目，seed 0–3，共 2000 条配对记录；不是 2000 道独立题目。", "",
             "![四状态分布](four_states.png)", "",
             "| 状态（Small, Large） | 含义 | 配对记录数 | 占比 |", "|---|---|---:|---:|"]
    for s in STATES:
        lines.append(f'| {s} | {LABELS[s]} | {pooled["counts"][s]} | {pooled["percentages"][s]:.2f}% |')
    lines += ["", f'小模型准确率 {pooled["small_accuracy_pct"]:.2f}%；大模型准确率 '
              f'{pooled["large_accuracy_pct"]:.2f}%；净增益 {pooled["large_minus_small_pp"]:.2f} 个百分点。', "",
              f'小模型错题中大模型答对比例：{pooled["counts"]["01"]} / '
              f'{pooled["counts"]["01"] + pooled["counts"]["00"]} = '
              f'{pooled["large_rescue_given_small_wrong_pct"]:.2f}%。',
              f'完整答案事后择优上界：{pooled["answer_selection_oracle_accuracy_pct"]:.2f}%。', "",
              "01 表示本次独立完整回答中仅大模型答对；00 只表示本次两份回答都错，不能证明永远无法解决。",
              "误差线是 seed 间样本标准差，不是置信区间。", "",
              "## 重建依据", "",
              "三位顺序为 (Small, Large, STEER)。原记录包含三者都对/都错、STEER 相对两组基线的配对得失，以及各自正确题数。",
              "由这些计数可唯一恢复八格列联表，再对 STEER 维度求和，得到 Small/Large 四状态。仅凭 Small/Large 准确率本身不能恢复。", "",
              "```text", "n110 = loses_small + loses_large − (N − all_wrong − steer_correct)",
              "n001 = rescues_small + rescues_large − (steer_correct − all_correct)",
              "n11 = all_correct + n110", "n00 = all_wrong + n001",
              "n10 = small_correct − n11", "n01 = large_correct − n11", "```", "",
              "原始归档输入及来源文件哈希见 archive_source.json。重建程序检查非负整数、总数与三组模型正确题数的一致性。",
              "没有逐题文件，不能恢复各状态题号、难度分布、完整回答或跨 seed 稳定性；本目录不生成这些明细。", "",
              "正式逐题结果请在服务器运行 experiment1_four_states.py；用其 report.md 与本预览交叉核对。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(pooled, ensure_ascii=False, indent=2))
    print(f"[归档预览完成] {output}")


if __name__ == "__main__":
    main()
