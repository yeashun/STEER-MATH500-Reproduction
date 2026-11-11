"""为专用环境中的 vLLM 安装官方 logits 补丁，支持备份与恢复。"""
import argparse
import importlib.metadata
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
FILES = ["sequence.py", "model_executor/layers/sampler.py"]


def apply_patch(root, restore=False):
    """幂等安装；原文件只备份一次，避免备份被补丁覆盖。"""
    # 修改前一次性检查全部源文件，避免恢复到一半才发现备份缺失。
    for rel in FILES:
        target = root / rel
        source = target.with_name(target.name + ".steer.bak") if restore else ROOT / "vllm_overwrite" / rel
        if not source.is_file() or not target.is_file():
            raise FileNotFoundError(f"补丁/备份/目标文件不存在: {source}, {target}")
    for rel in FILES:
        target = root / rel
        backup = target.with_name(target.name + ".steer.bak")
        if restore:
            shutil.copy2(backup, target)
        else:
            if target.read_bytes() == (ROOT / "vllm_overwrite" / rel).read_bytes():
                print("已安装:", rel)
                continue
            if not backup.exists():
                shutil.copy2(target, backup)
            shutil.copy2(ROOT / "vllm_overwrite" / rel, target)
        print("已恢复:" if restore else "已安装:", rel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    distribution = importlib.metadata.distribution("vllm")
    if distribution.version != "0.8.5.post1":
        raise RuntimeError("补丁只适用于 vllm==0.8.5.post1，不可覆盖其他版本")
    apply_patch(Path(distribution.locate_file("vllm")), args.restore)


if __name__ == "__main__":
    main()
