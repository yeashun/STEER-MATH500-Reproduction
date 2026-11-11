"""MATH500 首个复现目标的统一入口：small / large / steer / all。

本文件只负责任务配置与执行，不实现路由；算法入口见 ours_main.py。
每组实验在独立子进程中运行，结束后释放 GPU，再运行下一组。
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from check_env import check_dataset, check_patch, check_runtime
from summarize import summarize

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["small", "large", "steer", "all"], default="all")
    parser.add_argument("--samples", type=int, default=50, help="前 N 题；-1 表示全部 500 题")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/smoke")
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--target-model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5, help="保留上游的无题目完成轮数停止条件")
    parser.add_argument("--draft-threshold", type=float, default=0.4)
    parser.add_argument("--target-threshold", type=float, default=0.4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--small-params", type=float, default=1.5e9, help="FLOPs 采用的标称参数量")
    parser.add_argument("--large-params", type=float, default=7e9)
    parser.add_argument("--dry-run", action="store_true", help="仅检查数据并打印命令，不加载依赖或模型")
    args = parser.parse_args()
    if args.samples != -1 and not 1 <= args.samples <= 500:
        parser.error("--samples 必须为 -1 或 1..500")
    if not 0 <= args.draft_threshold <= 1 or not 0 <= args.target_threshold <= 1:
        parser.error("两个后验阈值必须位于 [0,1]")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("temperature 必须非负，top-p 必须位于 (0,1]")
    if args.max_tokens < 2 or args.max_model_len < 2 * args.max_tokens:
        parser.error("max-tokens 至少为 2；max-model-len 至少为 max-tokens 的两倍")
    if args.max_steps < 1 or args.patience < 1:
        parser.error("max-steps 和 patience 必须为正整数")
    if not 0 < args.gpu_memory_utilization < 1 or min(args.small_params, args.large_params) <= 0:
        parser.error("显存比例必须位于 (0,1)，参数量必须为正数")
    args.output = args.output.resolve()
    # 本地权重路径相对于调用者目录解析，避免子进程切换 cwd 后找不到。
    for name in ("draft_model", "target_model"):
        value = getattr(args, name)
        if Path(value).exists():
            setattr(args, name, str(Path(value).resolve()))
    return args


def build_command(args, method, directory):
    """全部使用参数列表调用，路径有中文或空格时也不用手工 shell 转义。"""
    if method == "steer":
        return [sys.executable, "-u", str(ROOT / "ours_main.py"),
                "--data_names", "math500", "--data_dir", str(ROOT / "data"),
                "--draft_model_name_or_path", args.draft_model,
                "--target_model_name_or_path", args.target_model,
                "--output_dir", str(directory), "--split", "test",
                "--prompt_type", "qwen25-math-cot", "--num_test_sample", str(args.samples),
                "--seed", str(args.seed), "--temperature", str(args.temperature),
                "--n_sampling", "1", "--top_p", str(args.top_p), "--save_outputs",
                "--max_tokens_per_call", str(args.max_tokens), "--max_steps", str(args.max_steps),
                "--patience", str(args.patience), "--use_step_reliability",
                "--reliability_metric", "R_eu", "--reliability_mode", "math_only_avg",
                "--reliability_k_top", "1", "--reliability_target_usage", "gmm_responsibility",
                "--draft_gmm_threshold", str(args.draft_threshold),
                "--target_gmm_threshold", str(args.target_threshold),
                "--gpu_memory_utilization", str(args.gpu_memory_utilization),
                "--max_model_len", str(args.max_model_len)]
    model = args.draft_model if method == "small" else args.target_model
    return [sys.executable, "-u", str(ROOT / "baseline.py"), "--method", method,
            "--model", model, "--output", str(directory), "--samples", str(args.samples),
            "--seed", str(args.seed), "--temperature", str(args.temperature),
            "--top-p", str(args.top_p), "--max-tokens", str(args.max_tokens),
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization)]


def termination_signal(returncode):
    """POSIX 子进程的负退出码表示信号；普通异常退出不应被误判为信号。"""
    if returncode >= 0:
        return None
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal {-returncode}"


def log_tail(path, lines=80, max_bytes=65536):
    """只读取日志末尾，避免失败时把整份推理日志刷到终端。"""
    with path.open("rb") as log:
        size = log.seek(0, os.SEEK_END)
        log.seek(max(0, size - max_bytes))
        content = log.read().decode("utf-8", errors="replace")
    return "\n".join(content.splitlines()[-lines:])


def show_and_save(lines, log):
    """同一份子进程输出同时显示到终端和写入完整日志。"""
    for line in lines:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)


def stop_interrupted_child(process, log):
    """只停止本次启动的子进程，并收集它退出前的剩余日志。"""
    if process.poll() is None:
        process.terminate()
    try:
        remaining, _ = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        remaining, _ = process.communicate()
    if remaining:
        show_and_save([remaining], log)
    return process.returncode


def execute(args, method, directory, command):
    """保存配置、完整日志与成功标记；失败任务不会被汇总成有效结果。"""
    directory.mkdir(parents=True, exist_ok=False)
    upstream = json.loads((ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"method": method, "upstream_commit": upstream["commit"],
                   "dataset_sha256": upstream["math500_sha256"],
                   "evaluation": "official_rule_based", "command": command,
                   "started_at_utc": datetime.now(timezone.utc).isoformat(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
                   "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "FASTEST_FIRST"),
                   # 记录请求的后端；实际采用的后端仍以 vLLM 初始化日志为准。
                   "attention_backend_requested": os.environ.get("VLLM_ATTENTION_BACKEND", "auto"),
                   "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING", "0"),
                   "budget_semantics": "upstream_steer_prompt_inclusive" if method == "steer" else "baseline_generated_tokens"})
    (directory / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    # 保存已安装环境和选用权重的 revision/文件配置，便于结果追溯。
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True)
    (directory / "environment.txt").write_text(frozen.stdout + frozen.stderr, encoding="utf-8")
    from transformers import AutoConfig
    selected = {"small": args.draft_model, "large": args.target_model}
    names = [args.draft_model, args.target_model] if method == "steer" else [selected[method]]
    revisions = {}
    for name in names:
        model_config = AutoConfig.from_pretrained(name)
        revisions[name] = {"resolved_revision": getattr(model_config, "_commit_hash", None),
                           "config": model_config.to_dict()}
    (directory / "model_configs.json").write_text(json.dumps(revisions, ensure_ascii=False, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update({"VLLM_USE_V1": "0", "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                "TOKENIZERS_PARALLELISM": "false", "PYTHONHASHSEED": str(args.seed)})
    started = time.perf_counter()
    interrupted = False
    print(f"开始 {method}；完整日志: {directory / 'run.log'}", flush=True)
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        # 正常推理不设置超时；只有用户 Ctrl+C 时才停止本次子进程。
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            try:
                config["child_pid"] = process.pid
                (directory / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
                show_and_save(process.stdout, log)
                returncode = process.wait()
            except KeyboardInterrupt:
                interrupted = True
                returncode = stop_interrupted_child(process, log)
    config["exit_code"] = returncode
    config["interrupted_by_user"] = interrupted
    config["termination_signal"] = termination_signal(returncode)
    config["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    config["process_seconds"] = time.perf_counter() - started
    (directory / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if interrupted:
        raise KeyboardInterrupt
    if returncode != 0:
        excerpt = log_tail(directory / "run.log")
        print("\n失败日志末尾（最多 80 行 / 64 KiB）：\n" + (excerpt or "[日志为空]"), file=sys.stderr, flush=True)
        reason = f"，信号 {config['termination_signal']}" if config["termination_signal"] else ""
        raise RuntimeError(f"{method} 失败，退出码 {returncode}{reason}；耗时 {config['process_seconds']:.1f} 秒；请检查 {directory / 'run.log'}")
    # 成功退出不够：再次核对题目数量、评测记录与 token 格式。
    from summarize import load_run
    load_run(directory)
    (directory / "SUCCESS").write_text("validated\n", encoding="utf-8")
    print(f"完成 {method}", flush=True)


def main():
    args = parse_args()
    check_dataset()
    methods = ["small", "large", "steer"] if args.method == "all" else [args.method]
    jobs = [(method, args.output / f"seed_{args.seed}" / method) for method in methods]
    for method, directory in jobs:
        if directory.exists() and not args.dry_run:
            raise FileExistsError(f"实验目录已存在: {directory}；请换 --output 或 --seed")
        print(f"\n[{method}] {shlex.join(build_command(args, method, directory))}", flush=True)
    if args.dry_run:
        print("\n预览结束：没有启动 GPU 推理，也没有创建结果目录。")
        return 0
    errors = check_runtime(2 if "steer" in methods else 1) + check_patch()
    if errors:
        raise RuntimeError("\n".join(errors))
    for method, directory in jobs:
        execute(args, method, directory, build_command(args, method, directory))
    summarize(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileExistsError) as error:
        print(f"运行中止: {error}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，运行已中断；已有输出保留。重试请使用新的 --output 目录。", file=sys.stderr)
        raise SystemExit(130)
