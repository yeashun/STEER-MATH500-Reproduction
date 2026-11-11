"""检查运行条件；不用下载模型，也不安装任何依赖。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent
EXPECTED_VERSIONS = {"vllm": "0.8.5.post1", "torch": "2.6.0", "transformers": "4.52.0"}


def check_dataset():
    """数量与哈希同时核对，防止误用别的数据版本。"""
    path = ROOT / "data/math500/test.jsonl"
    expected = json.loads((ROOT / "UPSTREAM.json").read_text(encoding="utf-8"))
    data = path.read_bytes()
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 500 or hashlib.sha256(data).hexdigest() != expected["math500_sha256"]:
        raise ValueError("MATH500 数量或 SHA256 与固定版本不一致")
    if len({row["unique_id"] for row in rows}) != 500:
        raise ValueError("MATH500 的 unique_id 重复")
    print("[OK] MATH500: 500 题，哈希与题号检查通过")


def check_runtime(required_gpus=2):
    """返回问题清单，让用户一次看到环境缺少什么。"""
    errors = []
    print(f"系统: {platform.system()}；Python: {platform.python_version()}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '[未设置：所有 CUDA 设备可见]')}")
    print(f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', 'FASTEST_FIRST（默认）')}")
    # 固定版本的 vLLM 在 NVML 路径中对每个设备标识调用 int()。
    # CUDA/PyTorch 支持 UUID，并不代表 vLLM 0.8.5.post1 也支持。
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    uuid_selection = any(value.strip().startswith(("GPU-", "MIG-"))
                         for value in visible_devices.split(","))
    if uuid_selection:
        errors.append("vLLM 0.8.5.post1 不兼容 UUID 形式的 CUDA_VISIBLE_DEVICES；"
                      "请使用数字索引，例如 GPU 7: export CUDA_DEVICE_ORDER=PCI_BUS_ID，"
                      "再执行 export CUDA_VISIBLE_DEVICES=7")
    if platform.system() != "Linux":
        errors.append("推理请在 Linux GPU 服务器或已验证 CUDA 的 WSL2 中执行")
    if sys.version_info[:2] != (3, 10):
        errors.append("请使用本项目固定的 Python 3.10 环境")
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
            # PyTorch 可能带 +cu124 等构建标记。
            if actual.split("+")[0] != expected:
                errors.append(f"{package}: 需要 {expected}，当前 {actual}")
            else:
                print(f"[OK] {package}: {actual}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"未安装 {package}=={expected}")
    if platform.system() == "Linux" and not uuid_selection:
        try:
            import torch
            count = torch.cuda.device_count()
            if count < required_gpus:
                errors.append(f"需要 {required_gpus} 张可见 GPU，当前 {count} 张")
            for index in range(min(count, required_gpus)):
                with torch.cuda.device(index):
                    if not torch.cuda.is_bf16_supported():
                        errors.append(f"GPU {index} 不支持本实验的 BF16 设置")
                    # 总容量不能代表可用空间；这里包含其他进程造成的占用。
                    free_memory, total_memory = torch.cuda.mem_get_info()
                prop = torch.cuda.get_device_properties(index)
                print(f"GPU {index}: {prop.name}；总显存 {total_memory / 2**30:.1f} GiB；"
                      f"当前可用 {free_memory / 2**30:.1f} GiB；"
                      f"已用 {(total_memory - free_memory) / 2**30:.1f} GiB")
                gpu_uuid = getattr(prop, "uuid", None)
                if gpu_uuid is not None:
                    print(f"  进程内 cuda:{index} 的 UUID: {gpu_uuid}")
        except (ImportError, RuntimeError, OSError) as error:
            errors.append(f"无法检查 CUDA: {error}")
    return errors


def check_patch():
    try:
        distribution = importlib.metadata.distribution("vllm")
        path = Path(distribution.locate_file("vllm/sequence.py"))
        if "top_logits_list" not in path.read_text(encoding="utf-8"):
            return ["vLLM 缺少 logits 补丁，请运行 python patch_vllm.py"]
        # 精确比较两个补丁文件，而非仅依据是否存在某个字段。
        for rel in ["sequence.py", "model_executor/layers/sampler.py"]:
            installed = Path(distribution.locate_file("vllm/" + rel))
            if installed.read_bytes() != (ROOT / "vllm_overwrite" / rel).read_bytes():
                return [f"vLLM 补丁不匹配: {rel}，请运行 python patch_vllm.py"]
        return []
    except (importlib.metadata.PackageNotFoundError, OSError) as error:
        return [f"无法核对 vLLM 补丁: {error}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true", help="只检查数据，Windows 也可运行")
    parser.add_argument("--gpus", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    check_dataset()
    if args.data_only:
        return 0
    errors = check_runtime(args.gpus) + check_patch()
    for error in errors:
        print("[待处理]", error)
    if not errors:
        print("[OK] 基础环境检查通过；显存是否足够仍须由小规模推理验证")
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
