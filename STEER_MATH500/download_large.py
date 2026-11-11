"""补齐独立目录中的 7B 权重：逐文件下载、网络失败重试，不读取旧全局缓存。

服务器运行：python download_large.py --directory models/Qwen2.5-Math-7B-Instruct_fresh_02
只操作指定模型目录；已完成文件和本次下载的临时分片继续使用。
"""
import argparse
import fnmatch
import json
import os
from pathlib import Path
import time

REPO = "Qwen/Qwen2.5-Math-7B-Instruct"
REVISION = "ef9926d75ab1d54532f6a30dd5e760355eb9aa4d"
PATTERNS = ("*.json", "*.safetensors", "*.model", "*.txt")


def retry_network(label, operation, retries, network_error):
    """ChunkedEncodingError 也属于 Requests 网络异常，立即重试当前文件。"""
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except network_error as error:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
            # 认证失败、文件不存在等错误不能靠反复重试解决。
            if status is not None and status < 500 and status not in (408, 429):
                raise
            print(f"[网络重试] {label}：{type(error).__name__}，第 {attempt}/{retries} 次", flush=True)
            if attempt == retries:
                raise
            time.sleep(5)


def download_files(directory, files, downloader, retries, network_error):
    """只在本次目录中续传；隔离缓存避免从旧全局缓存复制缺失文件。"""
    isolated_cache = directory / ".cache" / "isolated-hub"
    for item in sorted(files, key=lambda row: row.rfilename):
        filename = item.rfilename
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"模型文件路径异常：{filename}")
        print(f"[文件] {filename}", flush=True)
        path = Path(retry_network(
            filename,
            lambda: downloader(
                repo_id=REPO, filename=filename, revision=REVISION,
                local_dir=str(directory), cache_dir=str(isolated_cache),
                force_download=False, token=False,
            ),
            retries, network_error,
        ))
        size = getattr(item, "size", None)
        if not path.is_file() or (size is not None and path.stat().st_size != size):
            raise RuntimeError(f"文件大小不符合模型元数据：{filename}")
        print(f"[就绪] {filename}：{path.stat().st_size:,} 字节", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=5, help="每个文件最多尝试次数")
    args = parser.parse_args()
    if args.retries < 1:
        parser.error("--retries 至少为 1")
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)

    # 必须先设置变量，再导入 Hub；仅影响本次 Python 进程。
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_CACHE"] = str(directory / ".cache" / "isolated-hub")

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import enable_progress_bars
    from requests.exceptions import RequestException

    enable_progress_bars()
    print(f"[目录] {directory}\n[版本] {REVISION}", flush=True)
    info = retry_network(
        "获取模型文件列表",
        lambda: HfApi().model_info(REPO, revision=REVISION, files_metadata=True,
                                  timeout=30, token=False),
        args.retries, RequestException,
    )
    if info.sha != REVISION:
        raise RuntimeError("模型返回的 commit 与指定版本不一致")
    files = [item for item in info.siblings
             if any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in PATTERNS)]
    download_files(directory, files, hf_hub_download, args.retries, RequestException)

    index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 4 or any(not (directory / name).is_file() for name in shards):
        raise RuntimeError("本次 7B 模型的四个权重分片尚未齐全")
    source = {"repo_id": REPO, "revision": REVISION,
              "files": {item.rfilename: getattr(item, "size", None) for item in files}}
    (directory / "steer_model_source.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"权重准备完成：{len(shards)} 个分片；已检查文件大小", flush=True)
    print(f"本地模型目录：{directory}", flush=True)


if __name__ == "__main__":
    main()
