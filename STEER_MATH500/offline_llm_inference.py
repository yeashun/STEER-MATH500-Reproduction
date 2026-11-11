"""Qwen-Math 的 vLLM 推理 worker，保留官方 BF16/eager 设置。"""
import ray
from vllm import LLM


@ray.remote(num_gpus=1, num_cpus=4)
class OfflineOpenAIWrapper:
    """一个 actor 占一张 GPU；STEER 创建两个 actor，各加载一个模型。"""
    def __init__(self, model_name_or_path, tensor_parallel_size=1,
                 gpu_memory_utilization=0.8, **llm_kwargs):
        self._llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            dtype="bfloat16",
            **llm_kwargs,
        )

    def generate(self, **kwargs):
        return self._llm.generate(**kwargs)
