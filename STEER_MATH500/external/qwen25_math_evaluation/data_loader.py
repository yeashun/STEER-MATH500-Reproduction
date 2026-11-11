"""读取固定的 MATH500 文件，不从网上自动替换数据集。"""
from pathlib import Path
from .utils import load_jsonl


def load_data(data_name, split, data_dir):
    if data_name != "math500" or split != "test":
        raise ValueError("Only math500/test is supported")
    examples = list(load_jsonl(Path(data_dir) / "math500/test.jsonl"))
    return [{"idx": i, **item} for i, item in enumerate(examples)]
