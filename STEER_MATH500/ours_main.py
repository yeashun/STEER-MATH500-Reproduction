"""Qwen-Math + MATH500 的 STEER 入口。

输入输出层仅支持本次复现；get_responses 保留官方路由与停止策略。
阅读顺序：parse_args → setup → main → get_responses → utils.route_prompts。
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import ray
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from offline_llm_inference import OfflineOpenAIWrapper
from implement_logit_measure import dirichlet_au_eu
from utils import get_step_reliability, fit_mixture_model, get_gmm_responsibility, route_prompts
from external.qwen25_math_evaluation.data_loader import load_data
from external.qwen25_math_evaluation.evaluate import evaluate
from external.qwen25_math_evaluation.parser import parse_ground_truth, parse_question, run_execute
from external.qwen25_math_evaluation.utils import construct_prompt, save_jsonl, set_seed, json_safe


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_names", choices=["math500"], default="math500")
    parser.add_argument("--data_dir", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--draft_model_name_or_path", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--target_model_name_or_path", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--output_dir", default="results/direct_steer")
    parser.add_argument("--prompt_type", choices=["qwen25-math-cot"], default="qwen25-math-cot")
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--num_test_sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--n_sampling", type=int, choices=[1], default=1)
    parser.add_argument("--num_shots", type=int, choices=[0], default=0)
    parser.add_argument("--max_tokens_per_call", type=int, default=4096)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--use_step_reliability", action="store_true")
    parser.add_argument("--reliability_metric", choices=["R_eu"], default="R_eu")
    parser.add_argument("--reliability_mode", choices=["math_only_avg"], default="math_only_avg")
    parser.add_argument("--reliability_k_top", type=int, choices=[1], default=1)
    parser.add_argument("--reliability_target_usage", choices=["gmm_responsibility"], default="gmm_responsibility")
    parser.add_argument("--draft_gmm_threshold", type=float, default=0.4)
    parser.add_argument("--target_gmm_threshold", type=float, default=0.4)
    args = parser.parse_args()
    if args.num_test_sample != -1 and not 1 <= args.num_test_sample <= 500:
        parser.error("num_test_sample must be -1 or 1..500")
    if not (0 <= args.draft_gmm_threshold <= 1 and 0 <= args.target_gmm_threshold <= 1):
        parser.error("GMM thresholds must be in [0, 1]")
    if args.temperature == 0:
        args.top_p = 1
    args.step_word = "\n\n"
    return args


def setup(args):
    """大小模型各占一张卡，与官方双 Ray actor 的执行方式相同。"""
    ray.init(num_gpus=2, log_to_driver=False)
    common = dict(tensor_parallel_size=1, gpu_memory_utilization=args.gpu_memory_utilization,
                  max_model_len=args.max_model_len)
    draft = OfflineOpenAIWrapper.remote(args.draft_model_name_or_path, **common)
    target = OfflineOpenAIWrapper.remote(args.target_model_name_or_path, **common)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model_name_or_path)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name_or_path)
    main(args, draft, target, draft_tokenizer, target_tokenizer)


def main(args, draft, target, draft_tokenizer, target_tokenizer):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "samples.jsonl"
    if result_path.exists() or (output / "samples_unscored.jsonl").exists():
        raise FileExistsError("已有结果；请使用新的输出目录，避免 GMM 分组改变")
    examples = load_data("math500", "test", args.data_dir)
    if args.num_test_sample > 0:
        examples = examples[:args.num_test_sample]
    samples, prompts = [], []
    for example in examples:
        question = parse_question(example, "math500")
        reference_cot, reference = parse_ground_truth(example, "math500")
        samples.append({**example, "question": question, "gt_cot": reference_cot, "gt": reference})
        prompts.append(construct_prompt({"question": question}, "math500", args))
    print(f"MATH500: {len(samples)} questions; output={output}", flush=True)
    print("First prompt:", prompts[0], flush=True)
    started = time.perf_counter()
    # PRM/client/tokenizer 位置保留 None，主循环签名方便对照官方实现。
    outputs, counts, turns, _, reliabilities, _ = get_responses(
        args, draft, target, None, draft_tokenizer, target_tokenizer, None,
        prompts, [sample["question"] for sample in samples])
    generation_seconds = time.perf_counter() - started
    if not all(len(values) == len(samples) for values in [outputs, counts, turns, reliabilities]):
        raise RuntimeError("题目与生成/计数/路由日志长度不一致")
    for index, sample in enumerate(samples):
        code = outputs[index] or ""
        prediction, _ = run_execute(None, code, args.prompt_type, "math500")
        sample.update({"code": [code], "pred": [prediction], "report": [None],
                       "token_counts": [counts[index]], "turn_info": [turns[index]],
                       "step_reliabilities": [reliabilities[index]]})
    # 先保留昂贵的推理输出，再评分；评分失败也不会丢失生成结果。
    save_jsonl(samples, output / "samples_unscored.jsonl")
    evaluated, metrics = evaluate("math500", args.prompt_type, samples=samples)
    save_jsonl(evaluated, result_path)
    metrics.update({"generation_seconds": generation_seconds, "evaluation": "official_rule_based"})
    (output / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, allow_nan=False), encoding="utf-8")


def get_responses(args, draft_client, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer, prompts, problems):
    """官方步骤路由循环。

    首步：小模型生成全部题，低置信首步由大模型替换。
    后续：根据上轮生成模型和分数决定本轮模型，不逐步丢弃重生成。
    返回完整文本、计费 token、模型使用轨迹与每步置信度。
    """
    outputs = [None] * len(prompts)
    token_counts = [(0, 0, 0) for _ in prompts] # (draft_tokens, target_tokens, discarded_draft_tokens)
    step_info = [[] for _ in prompts] # List to store (step_num, client_id)
    all_rewards = [[] for _ in prompts] # (unused for now)
    final_step_reliabilities = [[] for _ in prompts] # List to store reliability info (step_num, metric_value, used_target) for each prompt
    detailed_token_logprobs = [[] for _ in prompts] # New: To store detailed token logprobs

    stop_sequences = [args.step_word, "<end_of_turn>", "<|im_end|>", "<|endoftext|>", "</s>"]
    stop_sequences_full_solve = ["<end_of_turn>", "<|im_end|>", "<|endoftext|>", "</s>"] # Stop at end of generation for full solve

    # 上游特殊回退分支：题目可被固定回小模型继续求解。
    idx_to_solve_with_draft = []

    print("--- Mode: STEER --- ")

    current_prompts = [(i, p, []) for i, p in enumerate(prompts)] # (index, base_prompt_text, previous_responses_list)
    num_step = 0
    num_unchanged = 0

    num_logprobs = 5 # not important
    sampling_params_step = SamplingParams(
        max_tokens=args.max_tokens_per_call,
        min_tokens=2,
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        logprobs=num_logprobs,
        stop=[args.step_word],
        include_stop_str_in_output=True,
        seed=args.seed
    )
    
    while current_prompts:
        prompts_for_draft_generation = [] 
        prompts_for_target_generation = [] 
        
        current_prompts_map = {item_tuple[0]: item_tuple for item_tuple in current_prompts}
        idx_value_pairs_draft = []
        idx_value_pairs_target = []
        for orig_idx, base_prompt_text, prev_responses_list in current_prompts:
            full_api_prompt = base_prompt_text + ''.join(r[0] for r in prev_responses_list)
            if num_step == 0: # first generate with draft model, then refine with target model
                prompts_for_draft_generation.append((orig_idx, full_api_prompt, prev_responses_list, base_prompt_text))
            else: # num_step > 0
                if final_step_reliabilities[orig_idx] and final_step_reliabilities[orig_idx][-1]: 
                    last_rel_info = final_step_reliabilities[orig_idx][-1]
                    decision_val = last_rel_info.get("decision_value", -float('inf'))
                    if last_rel_info.get("client_id") == 1:
                        idx_value_pairs_draft.append((orig_idx, decision_val))
                    elif last_rel_info.get("client_id") == 2:
                        idx_value_pairs_target.append((orig_idx, decision_val))
                else: 
                    if orig_idx in current_prompts_map: #
                        item_3_tuple = current_prompts_map[orig_idx]
                        o, b, p_list = item_3_tuple
                        full_api_p = b + ''.join(r[0] for r in p_list)
                        prompts_for_target_generation.append((o, full_api_p, p_list, b))
        
        if num_step != 0 and (idx_value_pairs_draft or idx_value_pairs_target):
            next_step_draft, next_step_target, target_rejection = route_prompts(args, idx_value_pairs_draft, idx_value_pairs_target, args.draft_gmm_threshold, args.target_gmm_threshold)
                                
            for idx in target_rejection:
                if idx not in idx_to_solve_with_draft:
                    idx_to_solve_with_draft.append(idx)
            
            temp_draft_prompts = []
            temp_target_prompts = []
            # Deduplicate by converting to set to avoid counting prompts multiple times
            next_step_draft = list(set(next_step_draft + idx_to_solve_with_draft))
            # Remove finished prompts from idx_to_solve_with_draft
            idx_to_solve_with_draft = [idx for idx in idx_to_solve_with_draft if idx in current_prompts_map]
            for idx in next_step_draft:
                if idx in current_prompts_map:
                    item_3_tuple = current_prompts_map[idx]
                    o, b, p_list = item_3_tuple
                    full_api_p = b + ''.join(r[0] for r in p_list)
                    temp_draft_prompts.append((o, full_api_p, p_list, b))
            for idx in next_step_target:
                if idx in idx_to_solve_with_draft:
                    continue
                item_3_tuple = current_prompts_map[idx]
                o, b, p_list = item_3_tuple
                full_api_p = b + ''.join(r[0] for r in p_list)
                temp_target_prompts.append((o, full_api_p, p_list, b))
            
            prompts_for_draft_generation = [p for p in prompts_for_draft_generation if p[0] not in next_step_draft]
            prompts_for_target_generation = [p for p in prompts_for_target_generation if p[0] not in next_step_target]
            prompts_for_draft_generation.extend(temp_draft_prompts)
            prompts_for_target_generation.extend(temp_target_prompts)
            
        current_step_generation_results = {} # Stores results for the current segment generation

        # print the routing info
        if num_step != 0:
            print(f"#### Step {num_step} - prompts to draft: {len(prompts_for_draft_generation)}, prompts to target: {len(prompts_for_target_generation)}")

        # 步骤 1：本轮被路由给小模型的题，生成一个双换行分隔的步骤。
        # --- Generate with Draft Model ---
        if prompts_for_draft_generation:
            # 下面代码与官方逐步生成路径一致。
            logical_step_texts = [[] for _ in range(len(prompts_for_draft_generation))]
            logical_step_raw_logprobs = [[] for _ in range(len(prompts_for_draft_generation))]
            logical_step_token_ids = [[] for _ in range(len(prompts_for_draft_generation))]
            logical_step_continue_generation = [True] * len(prompts_for_draft_generation)
            
            batch_texts_draft = [item[1] for item in prompts_for_draft_generation]
            step_newline_count = 1
            for _ in range(step_newline_count): # we are counting every double newline as a new step
                new_batch_texts_draft = []
                for i in range(len(batch_texts_draft)):
                    if not logical_step_continue_generation[i]:
                        continue
                    prev_responses = batch_texts_draft[i]
                    if not prev_responses.endswith(args.step_word):
                        prev_responses += args.step_word
                    current_generation = "".join(logical_step_texts[i])
                    new_prompt = prev_responses + current_generation
                    
                    # check if the new prompt is too long
                    tok_new_prompt = draft_tokenizer.encode(new_prompt)
                    if len(tok_new_prompt) >= args.max_model_len: # with some buffer
                        logical_step_continue_generation[i] = False
                        continue
                    
                    new_batch_texts_draft.append(new_prompt)
                
                draft_vllm_outputs = ray.get(draft_client.generate.remote(prompts=new_batch_texts_draft, sampling_params=sampling_params_step, use_tqdm=False))
                
                gen_idx = 0
                for i, (orig_idx, _, prev_responses, _) in enumerate(prompts_for_draft_generation):
                    if not logical_step_continue_generation[i]:
                        continue
                    vllm_output = draft_vllm_outputs[gen_idx].outputs[0]
                    logical_step_texts[i].append(vllm_output.text)
                    logical_step_raw_logprobs[i].extend(vllm_output.logprobs)
                    logical_step_token_ids[i].extend(vllm_output.token_ids)
                    
                    if ("\\boxed" in vllm_output.text) and vllm_output.text.endswith(args.step_word):
                        logical_step_continue_generation[i] = False
                    
                    gen_idx += 1
                
            for i, (orig_idx, _, prev_responses, _) in enumerate(prompts_for_draft_generation):
                response_text = "".join(logical_step_texts[i])
                raw_logprobs = logical_step_raw_logprobs[i]
                token_ids = logical_step_token_ids[i]
                
                step_token_strings = []
                stats_for_reliability_calc = []
                if raw_logprobs:
                    for d_token_logprob in raw_logprobs:
                        actual_logprob_obj = list(d_token_logprob.values())[0]
                        step_token_strings.append(actual_logprob_obj.decoded_token) 
                        if d_token_logprob and (args.reliability_metric == "R" or args.reliability_metric == "R_eu"):
                            if hasattr(actual_logprob_obj, 'top_logits_list') and actual_logprob_obj.top_logits_list is not None:
                                stats_for_reliability_calc.append(actual_logprob_obj.top_logits_list)
                            else:
                                raise RuntimeError("Missing top_logits_list; run python patch_vllm.py")
                        elif d_token_logprob and (args.reliability_metric == "entropy" or args.reliability_metric == "maxprob"):
                            logprobs = torch.tensor([obj.logprob for obj in d_token_logprob.values()])
                            stats_for_reliability_calc.append(logprobs)
                
                token_list_for_decision_metric = []  # 缺少 logits 时不能沿用上条样本变量。
                token_reliabilities_R_eu_list = []
                if stats_for_reliability_calc:
                    if args.reliability_metric == "R_eu":
                        for logit_tensor_list_item in stats_for_reliability_calc:
                            logit_tensor = torch.tensor(logit_tensor_list_item)
                            if logit_tensor.dim() == 0: continue
                            if logit_tensor.dim() > 1: logit_tensor = logit_tensor.squeeze()
                            if logit_tensor.dim() != 1: continue

                            au_eu_result = dirichlet_au_eu(logit_tensor.float(), args.reliability_k_top, only_eu=False) # only_eu=True means we only use the eu value
                            token_reliabilities_R_eu_list.append(1.0 / (au_eu_result["eu"] + 1e-9))
                            
                        token_list_for_decision_metric = token_reliabilities_R_eu_list
                    
                current_step_decision_value = get_step_reliability(
                    tokens=step_token_strings,
                    token_reliabilities_list=token_list_for_decision_metric,
                    mode=args.reliability_mode,
                )
                if current_step_decision_value is not None:
                    current_step_decision_value = float(current_step_decision_value)
                else:
                    current_step_decision_value = -float('inf') # Default if None

                token_details_for_this_step = []
                if step_token_strings:
                    for t_idx in range(len(step_token_strings)):
                        r_eu_val = token_reliabilities_R_eu_list[t_idx] if t_idx < len(token_reliabilities_R_eu_list) else None
                        token_details_for_this_step.append({
                            "token": step_token_strings[t_idx],
                            "R_eu": r_eu_val
                        })

                current_step_generation_results[orig_idx] = {
                    "text": response_text, "token_ids": token_ids, "stop_reason_raw": vllm_output.stop_reason,
                    "client_id": 1, 
                    "prev_responses": prev_responses,
                    "decision_value": current_step_decision_value # Store the decision value
                }

        if num_step == 0:
            actual_prompts_for_target_rerun = [] 
            
            temp_draft_generation_keys = [item[0] for item in prompts_for_draft_generation] 

            idx_value_pairs = []
            for orig_idx in temp_draft_generation_keys:
                if orig_idx in current_step_generation_results:
                    draft_gen_data = current_step_generation_results[orig_idx]
                    idx_value_pairs.append((orig_idx, draft_gen_data["decision_value"]))
            
            if idx_value_pairs and args.reliability_target_usage == "gmm_responsibility":
                print("GMM responsibility logic: First step rerun with target model")
                values = [v for _, v in idx_value_pairs]
                
                _, _, _, gmm_model = fit_mixture_model(values)
                _, classified_as_wrong_like, _ = get_gmm_responsibility(idx_value_pairs, gmm_model, args.draft_gmm_threshold, target_gmm=False)

                if classified_as_wrong_like:
                    for idx in classified_as_wrong_like:
                        prompt_to_rerun = next((p_tuple for p_tuple in prompts_for_draft_generation if p_tuple[0] == idx), None)
                        
                        orig_idx = prompt_to_rerun[0]
                        num_rejected_draft_tokens = len(current_step_generation_results[orig_idx]["token_ids"])
                        token_counts[orig_idx] = (
                            token_counts[orig_idx][0],
                            token_counts[orig_idx][1],
                            token_counts[orig_idx][2] + num_rejected_draft_tokens
                        )
                        
                        if prompt_to_rerun:
                            actual_prompts_for_target_rerun.append(prompt_to_rerun)
                        else:
                            print(f"[Warning] Could not find original prompt details for orig_idx {idx} during gmm_responsibility reject.")
                                    
            if actual_prompts_for_target_rerun:
                prompts_for_target_generation.extend(actual_prompts_for_target_rerun) 

                print(f"Re-running {len(actual_prompts_for_target_rerun)} prompts with target model (Reliability based)")

        # 步骤 3：首步补生成或后续升级的题，由大模型生成。
        # --- Generate with Target Model --- (No "all modes" analysis for target, just standard)
        if prompts_for_target_generation:
            batch_texts_target = [item[1] for item in prompts_for_target_generation]

            logical_step_texts = [[] for _ in range(len(prompts_for_target_generation))]
            logical_step_raw_logprobs = [[] for _ in range(len(prompts_for_target_generation))]
            logical_step_token_ids = [[] for _ in range(len(prompts_for_target_generation))]
            logical_step_continue_generation = [True] * len(prompts_for_target_generation)
            
            step_newline_count = 1
            for _ in range(step_newline_count): # we are counting every double newline as a new step
                new_batch_texts_target = []
                for i in range(len(batch_texts_target)):
                    if not logical_step_continue_generation[i]:
                        continue
                    prev_responses = batch_texts_target[i]
                    if not prev_responses.endswith(args.step_word):
                        prev_responses += args.step_word
                    current_generation = "".join(logical_step_texts[i])
                    new_prompt = prev_responses + current_generation
                    tok_new_prompt = target_tokenizer.encode(new_prompt)
                    if len(tok_new_prompt) >= args.max_model_len: # with some buffer
                        logical_step_continue_generation[i] = False
                        continue
                    new_batch_texts_target.append(new_prompt)
                    
                target_vllm_outputs = ray.get(target_client.generate.remote(prompts=new_batch_texts_target, sampling_params=sampling_params_step, use_tqdm=False))
                gen_idx = 0
                for i, (orig_idx, _, prev_responses, _) in enumerate(prompts_for_target_generation):
                    if not logical_step_continue_generation[i]:
                        continue
                    vllm_output = target_vllm_outputs[gen_idx].outputs[0]
                    logical_step_texts[i].append(vllm_output.text)
                    logical_step_raw_logprobs[i].extend(vllm_output.logprobs)
                    logical_step_token_ids[i].extend(vllm_output.token_ids)
                    
                    if ("\\boxed" in vllm_output.text) and vllm_output.text.endswith(args.step_word):
                        logical_step_continue_generation[i] = False
                    
                    gen_idx += 1
                
                for i, (orig_idx, _, prev_responses, _) in enumerate(prompts_for_target_generation):                        
                    response_text = "".join(logical_step_texts[i])
                    raw_logprobs = logical_step_raw_logprobs[i]
                    token_ids = logical_step_token_ids[i]

                    step_token_strings_target = [] 
                    step_logits_for_reliability_calc_target = [] 
                    if raw_logprobs:
                        for d_token_logprob in raw_logprobs:
                            actual_logprob_obj = list(d_token_logprob.values())[0] 
                            step_token_strings_target.append(actual_logprob_obj.decoded_token) 
                            if d_token_logprob and (args.reliability_metric == "R" or args.reliability_metric == "R_eu"):
                                if hasattr(actual_logprob_obj, 'top_logits_list') and actual_logprob_obj.top_logits_list is not None:
                                    step_logits_for_reliability_calc_target.append(actual_logprob_obj.top_logits_list)
                                else:
                                    raise ValueError("No top_logits_list found in vLLM Logprob object for target model")
                            elif d_token_logprob and (args.reliability_metric == "entropy" or args.reliability_metric == "maxprob"):
                                logprobs = torch.tensor([obj.logprob for obj in d_token_logprob.values()])
                                step_logits_for_reliability_calc_target.append(logprobs)
                    
                    token_list_for_decision_metric_target = []
                    token_reliabilities_R_eu_list_target = []
                    if step_logits_for_reliability_calc_target:
                        if args.reliability_metric == "R_eu":
                            for logit_tensor_list_item in step_logits_for_reliability_calc_target:
                                logit_tensor = torch.tensor(logit_tensor_list_item)
                                if logit_tensor.dim() == 0: continue
                                if logit_tensor.dim() > 1: logit_tensor = logit_tensor.squeeze()
                                if logit_tensor.dim() != 1: continue
                                au_eu_result = dirichlet_au_eu(logit_tensor.float(), args.reliability_k_top, only_eu=True) 
                                token_reliabilities_R_eu_list_target.append(1.0 / (au_eu_result["eu"] + 1e-9))
                            token_list_for_decision_metric_target = token_reliabilities_R_eu_list_target
                                
                    current_step_decision_value_target = get_step_reliability(
                        tokens=step_token_strings_target,
                        token_reliabilities_list=token_list_for_decision_metric_target,
                        mode=args.reliability_mode,
                    )
                    
                    if current_step_decision_value_target is not None:
                        current_step_decision_value_target = float(current_step_decision_value_target)
                    else:
                        current_step_decision_value_target = -float('inf')
                    
                    token_details_for_this_target_step = []
                    if step_token_strings_target:
                        for t_idx in range(len(step_token_strings_target)):
                            r_eu_val = token_reliabilities_R_eu_list_target[t_idx] if t_idx < len(token_reliabilities_R_eu_list_target) else None
                            token_details_for_this_target_step.append({
                                "token": step_token_strings_target[t_idx],
                                "R_eu": r_eu_val
                            })

                    current_step_generation_results[orig_idx] = {
                        "text": response_text, "token_ids": token_ids, "stop_reason_raw": vllm_output.stop_reason,
                        "client_id": 2, 
                        "prev_responses": prev_responses,
                        "decision_value": current_step_decision_value_target
                    }
        
        # 步骤 4：更新轨迹、计费 token、停止状态，下一轮用本轮分数做路由。
        next_prompts_temp = []
        num_finished_this_step = 0
        
        for orig_idx, base_prompt_text, _ in current_prompts: 
            if orig_idx not in current_step_generation_results:
                print(f"Warning: No generation result for index {orig_idx} in step {num_step}. Marking as finished to avoid loop.")
                if outputs[orig_idx] is None: outputs[orig_idx] = "ERROR_NO_GENERATION_RESULT_FOR_STEP"
                num_finished_this_step +=1 
                continue

            data = current_step_generation_results[orig_idx] 
            response_text_from_data = data["text"] # Use a different variable name to avoid confusion
            client_id = data["client_id"] 
            prev_responses_list = data["prev_responses"] 
            num_tokens = len(data["token_ids"]) 

            if client_id == 1: 
                token_counts[orig_idx] = (token_counts[orig_idx][0] + num_tokens, token_counts[orig_idx][1], token_counts[orig_idx][2])
            else: 
                token_counts[orig_idx] = (token_counts[orig_idx][0], token_counts[orig_idx][1] + num_tokens, token_counts[orig_idx][2])

            step_info[orig_idx].append((num_step, client_id))
            
            final_step_reliabilities[orig_idx].append({
                "step": num_step,
                "used_target": client_id == 2, 
                "client_id": client_id,
                "decision_value": data.get("decision_value", -float('inf')) # Ensure decision_value is stored
            })

            stop_reason_raw = data["stop_reason_raw"]
            mapped_stop_reason = "stop_sequence" 
            if stop_reason_raw == args.step_word: mapped_stop_reason = "step_word"
            elif stop_reason_raw == "length": mapped_stop_reason = "length"
            elif stop_reason_raw == "stop": mapped_stop_reason = "stop_sequence" 

            processed_response_text_for_step = response_text_from_data
            if mapped_stop_reason == "step_word":
                if not processed_response_text_for_step.endswith(args.step_word): 
                    processed_response_text_for_step += args.step_word
            elif mapped_stop_reason != "length" and not any(processed_response_text_for_step.strip().endswith(sw) for sw in stop_sequences_full_solve):
                    if not processed_response_text_for_step.endswith(args.step_word):
                        processed_response_text_for_step += args.step_word
            
            full_responses_updated_list = prev_responses_list + [(processed_response_text_for_step, client_id)]
            current_solution_text_only = ''.join(r[0] for r in full_responses_updated_list)
            full_text_for_len_check = base_prompt_text + current_solution_text_only

            max_tokens_reached = (len(draft_tokenizer.encode(full_text_for_len_check)) >= args.max_tokens_per_call or
                                    (target_tokenizer and len(target_tokenizer.encode(full_text_for_len_check)) >= args.max_tokens_per_call))

            is_finished = (("\\boxed" in processed_response_text_for_step) and processed_response_text_for_step.endswith(args.step_word) or 
                            (mapped_stop_reason == "length") or
                            max_tokens_reached or
                            (num_step >= args.max_steps - 1) or
                            any(current_solution_text_only.strip().endswith(sw) for sw in stop_sequences_full_solve))

            if is_finished:
                num_finished_this_step += 1
                outputs[orig_idx] = current_solution_text_only.strip()
                if outputs[orig_idx].endswith(args.step_word): 
                    outputs[orig_idx] = outputs[orig_idx][:-len(args.step_word)]
            else:
                next_prompts_temp.append((orig_idx, base_prompt_text, full_responses_updated_list))
        
        if num_finished_this_step > 0 or not next_prompts_temp: 
                if len(current_prompts) == len(next_prompts_temp) and num_finished_this_step == 0 : 
                    num_unchanged +=1
                else:
                    num_unchanged = 0 
        else: 
            num_unchanged += 1
        
        current_prompts = next_prompts_temp
        num_completed = len([o for o in outputs if o is not None])

        # Adjust print message slightly for clarity on mode
        print(f"#### Step {num_step}: Completed {num_completed} / {len(prompts)}, #unchanged {num_unchanged} / {args.patience}")
        num_step += 1

        if not current_prompts: 
            break
        if num_unchanged >= args.patience:
            print(f"Patience {args.patience} exceeded. Stopping generation for remaining {len(current_prompts)} prompts.")
            for orig_idx, base_prompt_text, prev_responses_list_patience in current_prompts: # Renamed to avoid clash
                if outputs[orig_idx] is None: 
                    final_text = ''.join(r[0] for r in prev_responses_list_patience).strip()
                    if final_text.endswith(args.step_word) and "\\boxed" not in args.step_word:
                            final_text = final_text[:-len(args.step_word)]
                    outputs[orig_idx] = final_text
                    
                    if orig_idx in current_step_generation_results: 
                        data_patience = current_step_generation_results[orig_idx] 
                        last_logged_step = -1
                        if final_step_reliabilities[orig_idx]:
                            last_logged_step = final_step_reliabilities[orig_idx][-1]['step']

                        if last_logged_step != (num_step - 1):
                            final_step_reliabilities[orig_idx].append({
                                "step": num_step - 1, 
                                # "R": data_patience.get("R"), "R_eu": data_patience.get("R_eu"),
                                "used_target": data_patience["client_id"] == 2,
                                "client_id": data_patience["client_id"],
                                "decision_value": data_patience.get("decision_value", -float('inf')),
                                "patience_exceeded_after_this_attempt": True
                            })
            current_prompts = [] 
            break 
        
    return outputs, token_counts, step_info, all_rewards, final_step_reliabilities, detailed_token_logprobs

if __name__ == "__main__":
    arguments = parse_args()
    set_seed(arguments.seed)
    try:
        setup(arguments)
    finally:
        ray.shutdown()
