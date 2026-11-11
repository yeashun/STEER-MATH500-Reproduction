"""MATH500 的 I/O 与官方 Qwen-Math 零样本提示词。"""
import os
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def json_safe(value):
    # 空数学步骤的置信度可能是 -inf，写文件时改为 JSON 标准支持的 null。
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def save_jsonl(samples, save_path):
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(json_safe(sample), ensure_ascii=False, allow_nan=False) + "\n")


PROMPT = '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n'


def construct_prompt(example, data_name, args):
    """与官方 qwen25-math-cot、num_shots=0 的最终 prompt 完全相同。"""
    if data_name != "math500" or args.prompt_type != "qwen25-math-cot" or args.num_shots != 0:
        raise ValueError("Only zero-shot Qwen-Math on MATH500 is supported")
    return PROMPT.format(input=example["question"]).strip(" ")
