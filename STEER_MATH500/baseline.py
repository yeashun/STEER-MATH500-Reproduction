"""单模型基线：一次生成完整解答，复用 STEER 的 prompt 和官方评测器。

只在真正开始推理时导入 GPU 依赖。CLI 帮助可在 Windows 上直接查看。
单模型使用一张 GPU；不把关闭 --use_step_reliability 当作单模型模式。
"""
import argparse
import json
from pathlib import Path
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["small", "large"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def prepare_model(model_name, revision=None):
    """先准备原始 safetensors 权重，再分配模型显存；复用已有 Hub 缓存。"""
    if Path(model_name).is_dir():
        return str(Path(model_name).resolve())

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import enable_progress_bars

    print(f"[阶段] 检查／下载模型文件：{model_name}；复用已有 Hugging Face 缓存", flush=True)
    if revision:
        print(f"[阶段] 固定模型 revision：{revision}", flush=True)
    # vLLM 内部使用 DisabledTqdm；这里先显式准备文件，显示 Hub 下载进度。
    enable_progress_bars()
    snapshot = Path(snapshot_download(
        repo_id=model_name, revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
        max_workers=2,
    ))
    if not (snapshot / "config.json").is_file() or not any(snapshot.glob("*.safetensors")):
        raise RuntimeError(f"模型快照缺少 config.json 或 safetensors 权重：{snapshot}")
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        missing = sorted(name for name in shards if not (snapshot / name).is_file())
        if missing:
            raise RuntimeError(f"模型快照缺少权重分片：{', '.join(missing)}")
    print(f"[阶段] 模型文件准备完成：{snapshot}", flush=True)
    return str(snapshot)


def run(args):
    # 启动器已保存 AutoConfig 的 commit；下载同一 revision，保证来源一致。
    metadata_path = args.output / "model_configs.json"
    configs = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    revision = configs.get(args.model, {}).get("resolved_revision")
    prepared_model = prepare_model(args.model, revision)
    args.output.mkdir(parents=True, exist_ok=True)
    source = {"requested_model": args.model, "requested_revision": revision,
              "prepared_model_path": prepared_model}
    (args.output / "model_source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")

    # 下载阶段完成之后才导入和初始化 GPU 推理引擎。
    from vllm import LLM, SamplingParams
    from external.qwen25_math_evaluation.data_loader import load_data
    from external.qwen25_math_evaluation.utils import construct_prompt, save_jsonl, set_seed, json_safe
    from external.qwen25_math_evaluation.parser import parse_ground_truth, run_execute
    from external.qwen25_math_evaluation.evaluate import evaluate

    set_seed(args.seed)
    root = Path(__file__).resolve().parent
    examples = load_data("math500", "test", root / "data")
    if args.samples > 0:
        examples = examples[:args.samples]
    # construct_prompt 的参数与官方 STEER 设置保持一致。
    args.prompt_type = "qwen25-math-cot"
    args.num_shots = 0
    prompts = [construct_prompt({"question": row["problem"].strip()}, "math500", args)
               for row in examples]
    print(f"[阶段] 数据与 prompt 准备完成：{len(prompts)} 题；开始加载模型 {args.model}", flush=True)
    loading_started = time.perf_counter()
    model = LLM(model=prepared_model, tensor_parallel_size=1, dtype="bfloat16",
                enforce_eager=True, gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len)
    print(f"[阶段] 模型加载完成，耗时 {time.perf_counter() - loading_started:.1f} 秒；开始生成", flush=True)
    started = time.perf_counter()
    # 与官方单模型 math_eval.py 一样，在 EOS 或完整解答上限处结束。
    outputs = model.generate(prompts, SamplingParams(
        temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        max_tokens=args.max_tokens, n=1,
        stop=["</s>", "<|im_end|>", "<|endoftext|>"],
        stop_token_ids=[151645, 151643],
    ))
    generation_seconds = time.perf_counter() - started
    print(f"[阶段] 生成完成，耗时 {generation_seconds:.1f} 秒", flush=True)
    if len(outputs) != len(examples):
        raise RuntimeError("模型输出数与输入题数不一致")
    samples = []
    for row, output in zip(examples, outputs):
        completion = output.outputs[0]
        prediction, _ = run_execute(None, completion.text, args.prompt_type, "math500")
        reference_cot, reference = parse_ground_truth(row, "math500")
        count = len(completion.token_ids)
        samples.append({
            **row, "gt_cot": reference_cot, "gt": reference,
            "code": [completion.text], "pred": [prediction], "report": [None],
            # 统一记为 (小模型保留 token, 大模型生成 token, 小模型丢弃 token)。
            "token_counts": [[count, 0, 0] if args.method == "small" else [0, count, 0]],
            "finish_reason": completion.finish_reason,
        })
    # 推理结果先落盘，即使符号评测失败，也能恢复这次昂贵的生成。
    save_jsonl(samples, args.output / "samples_unscored.jsonl")
    print("[阶段] 未评分结果已保存；开始数学答案评测", flush=True)
    evaluated, metrics = evaluate("math500", args.prompt_type, samples=samples)
    save_jsonl(evaluated, args.output / "samples.jsonl")
    metrics["generation_seconds"] = generation_seconds
    metrics["evaluation"] = "official_rule_based"
    (args.output / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, allow_nan=False), encoding="utf-8")
    print("[阶段] 评分结果与指标已保存", flush=True)


if __name__ == "__main__":
    run(parse_args())
