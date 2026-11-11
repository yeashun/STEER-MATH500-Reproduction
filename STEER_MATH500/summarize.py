"""从逐题结果重新核算准确率、生成 FLOPs 和三组结果；不依赖 torch。"""
import argparse
import csv
import json
from pathlib import Path

PAPER = {"small": (73.0, 1.82), "large": (79.6, 9.10), "steer": (79.6, 6.38)}


def calculate(samples, small_params=1.5e9, large_params=7e9):
    """按论文 2N/生成 token 近似，包含首步丢弃的小模型 token。"""
    if not samples:
        raise ValueError("没有逐题结果")
    ids = [sample["idx"] for sample in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("结果中题号重复，不能直接汇总")
    used_small = used_large = discarded_small = correct = empty = 0
    for sample in samples:
        if len(sample.get("score", [])) != 1 or len(sample.get("token_counts", [])) != 1:
            raise ValueError("首个复现目标要求每题一次采样，并有评测与 token 记录")
        a, b, c = sample["token_counts"][0]
        if any(not isinstance(v, int) or v < 0 for v in (a, b, c)):
            raise ValueError("token 数必须为非负整数")
        used_small += a
        used_large += b
        discarded_small += c
        correct += bool(sample["score"][0])
        empty += not sample.get("pred", [None])[0]
    count = len(samples)
    cost = (2 * small_params * (used_small + discarded_small)
            + 2 * large_params * used_large) / count / 1e12
    return {"num_questions": count, "correct": correct, "rule_accuracy": 100 * correct / count,
            "mean_flops_t": cost, "accuracy_per_flops": (100 * correct / count / cost) if cost else None,
            "small_used_tokens": used_small, "small_discarded_tokens": discarded_small,
            "large_generated_tokens": used_large, "empty_answers": empty}


def load_run(directory):
    metadata = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    # 不把保存于评测前的 samples_unscored.jsonl 重复计入结果。
    candidates = [path for path in directory.rglob("*.jsonl")
                  if path.name != "samples_unscored.jsonl"]
    if len(candidates) != 1:
        raise ValueError(f"{directory}: 应只有一个已评测结果文件，当前 {len(candidates)}")
    samples = [json.loads(line) for line in candidates[0].read_text(encoding="utf-8").splitlines()
               if line.strip()]
    expected = 500 if metadata["samples"] == -1 else metadata["samples"]
    if len(samples) != expected:
        raise ValueError(f"{directory}: 预期 {expected} 题，实际 {len(samples)}")
    row = calculate(samples, metadata["small_params"], metadata["large_params"])
    row.update({"method": metadata["method"], "seed": metadata["seed"],
                "run_directory": str(directory), "evaluation": "official_rule_based"})
    row["paper_accuracy"], row["paper_flops_t"] = PAPER[row["method"]]
    return row, metadata, [sample["idx"] for sample in samples]


def summarize(root):
    runs = [load_run(path.parent) for path in sorted(root.rglob("run_config.json"))
            if (path.parent / "SUCCESS").exists()]
    if not runs:
        raise ValueError("没有已完成实验；请检查 run.log。未完成实验不会参与统计。")
    # 只有同一个 seed、相同题目和设置的组，才能计算相对大模型成本。
    fields = ["samples", "draft_model", "target_model", "temperature", "top_p", "max_tokens",
              "max_model_len", "small_params", "large_params", "dataset_sha256"]
    for row, config, ids in runs:
        row["saving_vs_large_pct"] = None
        for large, other, other_ids in runs:
            if large["method"] == "large" and large["seed"] == row["seed"]:
                if ids == other_ids and all(config[k] == other[k] for k in fields):
                    if large["mean_flops_t"]:
                        row["saving_vs_large_pct"] = 100 * (1 - row["mean_flops_t"] / large["mean_flops_t"])
                    break
    rows = [row for row, _, _ in runs]
    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("method   seed  questions  rule_acc(%)  FLOPs(10^12)  saving_vs_large(%)")
    for row in rows:
        saving = "-" if row["saving_vs_large_pct"] is None else f'{row["saving_vs_large_pct"]:.1f}'
        print(f'{row["method"]:8} {row["seed"]:4} {row["num_questions"]:10} '
              f'{row["rule_accuracy"]:12.1f} {row["mean_flops_t"]:13.3f} {saving:>19}')
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    summarize(args.results.resolve())
