# 环境、运行与统计复核

## 安装与短流程

GPU 推理采用 Linux + Python 3.10，STEER 的两个 Ray actor 各占一张支持 BF16 的 GPU；small、large 基线各占一张。以下为 Bash 命令。

```bash
cd STEER_MATH500
conda create -n steer_math500 python=3.10 -y
conda activate steer_math500
python -m pip install -r requirements.txt
python -m pip check
python patch_vllm.py
export VLLM_USE_V1=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 填写分配给自己的两张物理 GPU 的数字编号；vLLM 0.8.5 不支持这里用 UUID。
export CUDA_VISIBLE_DEVICES=0,1
python check_env.py --gpus 2
python run.py --method all --samples 10 --seed 0 --output results/smoke_01
python run.py --method steer --samples 50 --seed 0 --output results/routing_check_01
```

依赖固定 vLLM 0.8.5.post1、PyTorch 2.6.0、Transformers 4.52.0。CUDA wheel、驱动与显存容量需要在实际服务器检查。10 题是接口检查，50 题用于观察 GMM 路由。输出目录必须是新目录；不要覆盖历史结果。补丁恢复：`python patch_vllm.py --restore`。

## 主实验协议

| 项目 | 设置 |
|---|---|
| 数据 | data/math500/test.jsonl，500 题，保持原顺序 |
| 数据 SHA256 | 55e02317c57e6c22edc3acf444702e0653acbef8a8f49bfc58b77d6bfefc5693 |
| 模型 | Qwen/Qwen2.5-Math-1.5B-Instruct 与 Qwen/Qwen2.5-Math-7B-Instruct |
| seed / 采样次数 | 0、1、2、3 / 每题每轮一次 |
| temperature / top-p | 0.7 / 1.0 |
| max-tokens / max-model-len | 4096 / 8192 |
| STEER max-steps / patience | 100 / 5 |
| 置信度 | R_eu，math_only_avg，k_top=1 |
| draft / target 阈值 | 低置信分量后验阈值均为 0.4 |
| 评测 | 仓库规则评测；未运行 xVerify |

在独立的新输出目录复跑：

```bash
for seed in 0 1 2 3; do
  python run.py --method all --samples -1 --seed "$seed"     --output results/reproduction_01 || break
done
python summarize.py results/reproduction_01
```

只有一张 GPU 时分别运行 `--method small` 和 `--method large`，先用 50 题检查。模型默认通过 Hub 获取，已有本地权重可传 `--draft-model` / `--target-model`；离线缓存完整时可为命令加 `HF_HUB_OFFLINE=1`。

## 论文描述与代码口径

首步统一由小模型生成，低置信时由大模型重生成；后续根据前一步的置信度选择下一步模型，不能把每步都描述成先小模型生成再重做。步骤由 `\n\n` 分隔。

论文描述最大 logit；官方实现采用 R_eu，k_top=1 时在通常正值分支近似为最大 logit + 1。补丁保留的是采样变换之后、softmax 之前的 logits；不要直接称作完全未经变换的 raw logits。

论文的高置信后验阈值 γ 与代码低置信后验阈值 τ 通常满足 τ=1−γ；边界不等式及回退分支仍以实现为准。发布代码包含后续修复；恢复服务器采集时三份不同运行模块的补丁见 provenance/recorded-code.patch，在仓库根目录的独立副本上用 `git -c core.autocrlf=false apply provenance/recorded-code.patch` 应用。

## 重算已发布的结果

仓库根目录执行 `python scripts/verify_published_results.py`，仅使用标准库，从精简后的逐题评分/token 表、四状态明细和代理值复核统计。CSV 保留计算所需字段，省略完整回答和重复内容。

分析脚本的完整输入还需要原始回答档案，未随此仓库发布。原工作目录中的 `reports/experiment1_remote_check_20261023_01/input_sources` 保留基线副本；服务器保留三方法全量结果。取回原始结果后，在 STEER_MATH500 目录执行：

```bash
python -m pip install -r requirements-analysis.txt
python experiment1_four_states.py   --small-root results/gpu7_small_full_01 --large-root results/gpu7_large_full_01   --seeds 0 1 2 3 --samples -1 --output reports/experiment1_new
```

实验 2 的 `--source` 输入要求同时包含原始 `input_sources` 和四状态分析 `analysis` 目录，具体结构见脚本帮助。它不是只凭发布摘要即可重新运行的概率置信度实验。
