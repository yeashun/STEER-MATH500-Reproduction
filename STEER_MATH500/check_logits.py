"""生成一条短回答，实际验证 logits 补丁，而非只检查文件已替换。"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    args = parser.parse_args()
    import os
    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams
    from implement_logit_measure import dirichlet_au_eu
    from external.qwen25_math_evaluation.utils import PROMPT
    import torch
    model = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
                max_model_len=1024, gpu_memory_utilization=0.8)
    completion = model.generate([PROMPT.format(input="What is 1 + 1?")],
                                SamplingParams(max_tokens=32, temperature=0.7,
                                               top_p=1, logprobs=5, seed=0))[0].outputs[0]
    if not completion.logprobs or len(completion.logprobs) != len(completion.token_ids):
        raise RuntimeError("生成 token 与 logits 长度不一致或为空")
    for record in completion.logprobs:
        info = next(iter(record.values()))
        values = getattr(info, "top_logits_list", None)
        if not values:
            raise RuntimeError("缺少 top_logits_list；请运行 python patch_vllm.py")
    values = torch.tensor(next(iter(completion.logprobs[0].values())).top_logits_list)
    result = dirichlet_au_eu(values.float(), 1)
    print(f"[OK] {len(completion.token_ids)} 个 token 均带有 logits")
    print(f"首个 token 最大 logit: {values.max().item():.5f}")
    print(f"首个 token R_eu: {1 / (result['eu'] + 1e-9):.5f}")
    print("生成文本:", completion.text)


if __name__ == "__main__":
    main()
