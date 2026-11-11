"""分析已有 MATH500 结果；不启动 GPU，不下载模型，不修改输入文件。

默认只分析主实验 seed 0–3。--regrade 从保存的 code 重新提取答案，
使用项目内同一规则评测器复核；这不是 xVerify。
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

from summarize import calculate

ROOT = Path(__file__).resolve().parent
METHODS = ("small", "large", "steer")


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def single(sample, field):
    value = sample.get(field)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"题号 {sample.get('idx')}: {field} 必须只有一个记录")
    return value[0]


def score(sample):
    value = single(sample, "score")
    if not isinstance(value, (bool, int)) or value not in (0, 1):
        raise ValueError(f"题号 {sample.get('idx')}: score 必须是布尔值或 0/1")
    return int(value)


def validate_samples(samples, dataset, method):
    """按题号匹配标准题目，防止重复、缺题和三个方法错位比较。"""
    indexed = {}
    for sample in samples:
        idx = sample.get("idx")
        if type(idx) is not int or idx not in dataset or idx in indexed:
            raise ValueError(f"{method}: 非法或重复题号 {idx}")
        canonical = dataset[idx]
        for field in ("problem", "answer", "unique_id"):
            if str(sample.get(field, "")).strip() != str(canonical[field]).strip():
                raise ValueError(f"{method} 题号 {idx}: {field} 与标准数据不一致")
        score(sample)
        if not isinstance(single(sample, "code"), str):
            raise ValueError(f"{method} 题号 {idx}: code 必须是文本")
        single(sample, "pred")
        counts = single(sample, "token_counts")
        if (not isinstance(counts, (list, tuple)) or len(counts) != 3
                or any(type(v) is not int or v < 0 for v in counts)):
            raise ValueError(f"{method} 题号 {idx}: token_counts 非法")
        if (method == "small" and (counts[1] or counts[2])) or (
                method == "large" and (counts[0] or counts[2])):
            raise ValueError(f"{method} 题号 {idx}: token 计数与方法不一致")
        if method == "steer":
            turns = single(sample, "turn_info")
            if not isinstance(turns, list):
                raise ValueError(f"steer 题号 {idx}: turn_info 非法")
            previous = -1
            for turn in turns:
                if (not isinstance(turn, (list, tuple)) or len(turn) != 2
                        or type(turn[0]) is not int or turn[0] <= previous
                        or type(turn[1]) is not int or turn[1] not in (1, 2)):
                    raise ValueError(f"steer 题号 {idx}: 路由轨迹非法")
                previous = turn[0]
        indexed[idx] = sample
    if set(indexed) != set(dataset):
        raise ValueError(f"{method}: 缺失题号 {sorted(set(dataset) - set(indexed))[:10]}")
    return indexed


def question_stats(sample, config, method):
    a, b, c = single(sample, "token_counts")
    code = single(sample, "code")
    result = {
        "score": score(sample), "empty_answer": not bool(single(sample, "pred")),
        "small_tokens": a, "large_tokens": b, "discarded_small_tokens": c,
        "flops_t": (2 * config["small_params"] * (a + c)
                    + 2 * config["large_params"] * b) / 1e12,
        "has_boxed_text": "\\boxed" in code,
        "generation_error_text": code.startswith("ERROR_NO_GENERATION_RESULT"),
        # 不根据无 boxed、长度接近上限等文本特征猜测停止原因。
        "finish_reason": sample.get("finish_reason", "not_recorded"),
        "steps": None, "small_steps": None, "large_steps": None, "switches": None,
        "route_trace": None,
        "patience_flag_recorded": None, "max_steps_boundary": None,
    }
    if method == "steer":
        turns = single(sample, "turn_info")
        clients = [turn[1] for turn in turns]
        reliabilities = single(sample, "step_reliabilities") if "step_reliabilities" in sample else []
        result.update({
            "steps": len(turns), "small_steps": clients.count(1),
            "large_steps": clients.count(2),
            "switches": sum(a != b for a, b in zip(clients, clients[1:])),
            "route_trace": ";".join(f"{step}:{'S' if client == 1 else 'L'}" for step, client in turns),
            "patience_flag_recorded": any(item.get("patience_exceeded_after_this_attempt", False)
                                          for item in reliabilities),
            "max_steps_boundary": bool(turns and turns[-1][0] >= config["max_steps"] - 1),
        })
    return result


def audit_run(root, seed, method, dataset, digest):
    directory = root / f"seed_{seed}" / method
    if not (directory / "SUCCESS").is_file():
        raise ValueError(f"实验未成功或目录不存在：{directory}")
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    if config["method"] != method or config["seed"] != seed or config["samples"] != -1:
        raise ValueError(f"{directory}: 方法、seed 或全量题数配置不符")
    if config.get("dataset_sha256") != digest:
        raise ValueError(f"{directory}: 数据集哈希与当前标准文件不符")
    rows = validate_samples(read_jsonl(directory / "samples.jsonl"), dataset, method)
    metrics = calculate(list(rows.values()), config["small_params"], config["large_params"])
    saved = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if abs(float(saved["acc"]) - round(metrics["rule_accuracy"], 1)) > 1e-7:
        raise ValueError(f"{directory}: 保存的 acc 与逐题 score 重算不一致")
    if saved.get("num_samples") != len(dataset):
        raise ValueError(f"{directory}: metrics.json 的题数不一致")
    log_path = directory / "run.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    observed = []
    for phrase, backend in (("Using Flash Attention backend", "FLASH_ATTN"),
                            ("Using XFormers backend", "XFORMERS")):
        if phrase in log:
            observed.append(backend)
    requested = config.get("attention_backend_requested", "not_recorded")
    if requested == "XFORMERS" or "XFORMERS" in observed:
        raise ValueError(f"{directory}: 主结果应排除 XFormers 验证运行")
    checks = {
        "method": method, "seed": seed, "directory": str(directory),
        "dataset_hash_matches": True, "question_ids_and_content_match": True,
        "saved_accuracy_matches": True,
        "attention_backend_requested": requested, "attention_backends_observed": observed,
        "backend_confirmed_by_log": observed == ["FLASH_ATTN"],
        "cuda_launch_blocking": config.get("cuda_launch_blocking", "not_recorded"),
        "budget_semantics": config.get("budget_semantics", "not_recorded"),
        "saved_rule_timeouts": saved.get("timeout_samples"),
        "patience_stop_events": [int(n) for n in re.findall(
            r"Patience \d+ exceeded\. Stopping generation for remaining (\d+) prompts", log)],
        "metrics": metrics,
    }
    stats = [question_stats(row, config, method) for row in rows.values()]
    checks["finish_reason_counts"] = dict(Counter(item["finish_reason"] for item in stats))
    checks["generation_error_answers"] = sum(item["generation_error_text"] for item in stats)
    if method == "steer":
        total_tokens = sum(metrics[k] for k in ("small_used_tokens", "small_discarded_tokens", "large_generated_tokens"))
        checks["routing"] = {
            "mean_steps_per_question": statistics.mean(item["steps"] for item in stats),
            "small_steps": sum(item["small_steps"] for item in stats),
            "large_steps": sum(item["large_steps"] for item in stats),
            "model_switches": sum(item["switches"] for item in stats),
            "questions_with_large_tokens": sum(single(row, "token_counts")[1] > 0 for row in rows.values()),
            "large_fraction_of_all_generated_tokens": metrics["large_generated_tokens"] / total_tokens if total_tokens else None,
            "discarded_fraction_of_all_generated_tokens": metrics["small_discarded_tokens"] / total_tokens if total_tokens else None,
            "explicit_patience_flags": sum(item["patience_flag_recorded"] for item in stats),
            "questions_at_max_steps_boundary": sum(item["max_steps_boundary"] for item in stats),
        }
    return {"method": method, "seed": seed, "config": config, "samples": rows,
            "checks": checks, "regrade_metrics": None}


def regrade_run(run, dataset, evaluator=None, extractor=None):
    """从 code 重提取答案，以标准题目重新评分；旧 pred/score 保持原样。"""
    if evaluator is None:
        from external.qwen25_math_evaluation.evaluate import evaluate
        from external.qwen25_math_evaluation.parser import run_execute
        evaluator, extractor = evaluate, run_execute
    inputs = []
    for idx, old in sorted(run["samples"].items()):
        prediction, _ = extractor(None, single(old, "code"), "qwen25-math-cot", "math500")
        inputs.append({**deepcopy(dataset[idx]), "idx": idx,
                       "code": deepcopy(old["code"]), "pred": [prediction], "report": [None]})
    try:
        scored, metrics = evaluator("math500", "qwen25-math-cot", samples=inputs)
    except SystemExit as error:
        raise RuntimeError("规则评测器中止，未完成评分核对") from error
    indexed = {item["idx"]: item for item in scored}
    if len(indexed) != len(scored) or set(indexed) != set(run["samples"]):
        raise ValueError("重评分返回题号重复或缺失")
    changes = []
    for idx, old in sorted(run["samples"].items()):
        new = indexed[idx]
        row = {"method": run["method"], "seed": run["seed"], "idx": idx,
               "old_pred": single(old, "pred"), "new_pred": single(new, "pred"),
               "old_score": score(old), "new_score": score(new),
               "old_gt": old.get("gt"), "new_gt": new.get("gt")}
        row.update({"prediction_changed": row["old_pred"] != row["new_pred"],
                    "score_changed": row["old_score"] != row["new_score"],
                    "reference_changed": row["old_gt"] != row["new_gt"]})
        if any(row[k] for k in ("prediction_changed", "score_changed", "reference_changed")):
            changes.append(row)
    run["regrade_metrics"] = metrics
    return changes


def mean_sd(values):
    return {"mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def paired_counts(rows):
    counts = Counter()
    for row in rows:
        small, large, steer = (row[f"{m}_score"] for m in METHODS)
        counts["steer_rescues_small"] += not small and steer
        counts["steer_loses_small"] += small and not steer
        counts["steer_loses_large"] += large and not steer
        counts["steer_rescues_large"] += not large and steer
        counts["all_wrong"] += not any((small, large, steer))
        counts["all_correct"] += all((small, large, steer))
    return dict(counts)


def write_csv(path, rows):
    if not rows:
        # 空文件也保留明确表头，便于查看“没有发现变化”。
        path.write_text("status\nno_records\n", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_report(runs, dataset, seeds, changes):
    indexed = {(run["seed"], run["method"]): run for run in runs}
    per_question, groups, review = [], [], []
    changed = {(row["seed"], row["method"], row["idx"]) for row in changes}
    pair_rows = []
    for seed in seeds:
        for idx, canonical in sorted(dataset.items()):
            row = {"seed": seed, "idx": idx, "unique_id": canonical["unique_id"],
                   "level": canonical.get("level"), "subject": canonical.get("subject"),
                   "question": canonical["problem"], "reference_answer": canonical["answer"]}
            samples = {}
            for method in METHODS:
                run = indexed[seed, method]
                sample = samples[method] = run["samples"][idx]
                stats = question_stats(sample, run["config"], method)
                row.update({f"{method}_{key}": value for key, value in stats.items()})
                row[f"{method}_pred"] = single(sample, "pred")
            per_question.append(row)
            reasons = []
            if row["large_score"] and not row["steer_score"]:
                reasons.append("large_correct_steer_wrong")
            if row["small_score"] and not row["steer_score"]:
                reasons.append("small_correct_steer_wrong")
            if not row["small_score"] and row["steer_score"]:
                reasons.append("steer_rescues_small")
            if any(row[f"{m}_empty_answer"] or row[f"{m}_generation_error_text"] for m in METHODS):
                reasons.append("empty_answer_or_error_text")
            if any((seed, m, idx) in changed for m in METHODS):
                reasons.append("regrade_changed")
            if reasons:
                review.append({**row, "review_reasons": reasons,
                               "steer_step_reliabilities": samples["steer"].get("step_reliabilities", [[]])[0],
                               "responses": {m: single(samples[m], "code") for m in METHODS}})
        seed_rows = [row for row in per_question if row["seed"] == seed]
        pair_rows.append({"seed": seed, **paired_counts(seed_rows)})
        for field in ("level", "subject"):
            grouped = defaultdict(list)
            for row in seed_rows:
                grouped[str(row[field])].append(row)
            for value, members in sorted(grouped.items()):
                for method in METHODS:
                    groups.append({"seed": seed, "method": method, "group_by": field,
                                   "group": value, "questions": len(members),
                                   "accuracy_pct": 100 * statistics.mean(r[f"{method}_score"] for r in members),
                                   "mean_flops_t": statistics.mean(r[f"{method}_flops_t"] for r in members)})
    summary = {}
    for method in METHODS:
        method_runs = [indexed[seed, method] for seed in seeds]
        metrics = [r["checks"]["metrics"] for r in method_runs]
        summary[method] = {"seeds": seeds, "questions_per_seed": len(dataset),
                           "accuracy_pct": mean_sd([m["rule_accuracy"] for m in metrics]),
                           "mean_flops_t": mean_sd([m["mean_flops_t"] for m in metrics]),
                           "empty_answers_per_seed": [m["empty_answers"] for m in metrics]}
    large_cost = summary["large"]["mean_flops_t"]["mean"]
    summary["steer"]["saving_from_ratio_of_mean_flops_pct"] = (
        100 * (1 - summary["steer"]["mean_flops_t"]["mean"] / large_cost) if large_cost else None)
    return summary, per_question, groups, pair_rows, review


def analyze(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"分析目录已存在，请换 --output：{output}")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("seed 必须非负且不能重复")
    data_path = ROOT / "data/math500/test.jsonl"
    dataset = {idx: {"idx": idx, **item} for idx, item in enumerate(read_jsonl(data_path))}
    if len(dataset) != 500:
        raise ValueError("标准 MATH500 文件必须包含 500 题")
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    runs, changes = [], []
    for seed in args.seeds:
        for method in METHODS:
            print(f"[核对] {method} seed={seed}", flush=True)
            run = audit_run(getattr(args, f"{method}_root").resolve(), seed, method, dataset, digest)
            runs.append(run)
            if args.regrade:
                changes.extend(regrade_run(run, dataset))
    for seed in args.seeds:
        selected = [r for r in runs if r["seed"] == seed]
        for key in ("temperature", "top_p", "max_tokens", "max_model_len", "small_params", "large_params"):
            if len({r["config"][key] for r in selected}) != 1:
                raise ValueError(f"seed={seed}: 三方法的 {key} 不一致，不能直接配对")
        # 只比较实际使用的模型；small 配置里的未使用 target 路径不参与。
        small, large, steer = selected
        if (small["config"]["draft_model"] != steer["config"]["draft_model"]
                or large["config"]["target_model"] != steer["config"]["target_model"]):
            raise ValueError("实际生成模型路径不同，需要先核对权重来源")
    for method in METHODS:
        method_runs = [r for r in runs if r["method"] == method]
        keys = ["draft_model", "target_model", "temperature", "top_p", "max_tokens", "max_model_len",
                "small_params", "large_params", "gpu_memory_utilization"]
        if method == "steer":
            keys += ["max_steps", "patience", "draft_threshold", "target_threshold"]
        for key in keys:
            if len({r["config"].get(key) for r in method_runs}) != 1:
                raise ValueError(f"{method}: 跨 seed 的 {key} 不一致")
    summary, questions, groups, pairs, review = make_report(runs, dataset, args.seeds, changes)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "per_question.csv", questions)
    write_csv(output / "by_level_subject.csv", groups)
    write_csv(output / "paired_outcomes.csv", pairs)
    write_csv(output / "regrade_changes.csv", changes)
    write_csv(output / "manual_review.csv", [{k: v for k, v in row.items()
                                                if k not in ("responses", "steer_step_reliabilities")}
                                               for row in review])
    with (output / "manual_review.jsonl").open("w", encoding="utf-8") as stream:
        for row in review:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    checks = {"dataset_sha256": digest, "selected_seeds": args.seeds,
              "regrade_requested": args.regrade, "runs": [r["checks"] for r in runs],
              "regrade_metrics": [{"method": r["method"], "seed": r["seed"],
                                     "metrics": r["regrade_metrics"]} for r in runs],
              "regrade_changed_records": len(changes),
              "limitations": [
                  "模型路径相同不等于权重文件已做内容哈希核验；本脚本不读取大权重。",
                  "请求后端与实际后端不同；无初始化日志的运行无法确认实际后端。",
                  "STEER 未完整保存逐题停止原因；未记录 patience 标记不能证明未触发。",
                  "STEER 的预算包含 prompt，baseline 的预算只限制生成 token，不能称为严格等预算对照。",
                  "重新评分使用同一规则评测器，不是独立正确性证明或 xVerify；超时可能造成变化。",
                  "同一题目的跨 seed 重复不构成独立新题目；标准差按 seed 均值计算。",
              ]}
    for name, data in (("checks.json", checks), ("summary.json", summary)):
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                  encoding="utf-8")
    lines = ["# MATH500 现有结果逐题核对", "", f"种子：{args.seeds}；每种方法每轮 500 题。",
             "原结果不修改。主表使用原 score；重评分变化单独列出。", "",
             "| 方法 | 准确率均值 (%) | seed 间样本标准差 (百分点) | 平均 FLOPs/题 (×10¹²) |",
             "|---|---:|---:|---:|"]
    for method in METHODS:
        item = summary[method]
        sd = item["accuracy_pct"]["sample_sd"]
        sd_text = f"{sd:.2f}" if sd is not None else "未定义"
        lines.append(f"| {method} | {item['accuracy_pct']['mean']:.2f} | "
                     f"{sd_text} | {item['mean_flops_t']['mean']:.6f} |")
    saving = summary["steer"]["saving_from_ratio_of_mean_flops_pct"]
    saving_text = f"{saving:.2f}%" if saving is not None else "未定义（large 的生成成本为 0）"
    lines += ["", f"STEER 相对 large 的均值 FLOPs 节省：{saving_text}。",
              "这是当前运行设置的近似计算量比较，不代表耗时、能耗或严格等预算比较。", "",
              f"重新评分：{'已执行' if args.regrade else '未执行'}；变化记录数：{len(changes)}。",
              "变化可能是答案提取、参考答案或分数变化，逐项查看 regrade_changes.csv。", "",
              "逐题表：per_question.csv；难度/学科表：by_level_subject.csv；配对得失：paired_outcomes.csv。",
              "优先人工复核：manual_review.csv；对应三种方法的完整回答：manual_review.jsonl。", "",
              "## 核对范围与限制", ""] + [f"- {item}" for item in checks["limitations"]]
    missing_backend = sum(not r["checks"]["backend_confirmed_by_log"] for r in runs)
    lines += ["", f"未从初始化日志确认 FlashAttention 的运行数：{missing_backend}，详情见 checks.json。",
              "日志记录的 patience 事件与题数见 checks.json；逐题停止原因缺失时保留 unknown，不从文本反推。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("method  accuracy_mean(%)  sample_sd(pp)  mean_flops(1e12)")
    for method in METHODS:
        item = summary[method]
        print(method, item["accuracy_pct"]["mean"], item["accuracy_pct"]["sample_sd"], item["mean_flops_t"]["mean"])
    print(f"[完成] 分析目录：{output}\n重评分变化记录：{len(changes)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-root", type=Path, default=ROOT / "results/gpu7_small_full_01")
    parser.add_argument("--large-root", type=Path, default=ROOT / "results/gpu7_large_full_01")
    parser.add_argument("--steer-root", type=Path, default=ROOT / "results/gpu12_steer_full_01")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regrade", action="store_true", help="CPU 重新提取 code 中的答案并复核规则评分")
    args = parser.parse_args()
    try:
        analyze(args)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as error:
        parser.exit(1, f"分析中止：{error}\n")


if __name__ == "__main__":
    main()
