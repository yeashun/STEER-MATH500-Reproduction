"""CPU 回归检查：统计、任务隔离、补丁恢复和裁剪后的本地导入。

运行 python -m unittest discover -s tests -v；不下载权重，不创建真实实验结果。
"""
import ast
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from summarize import calculate, summarize
from patch_vllm import apply_patch
from run import execute, log_tail, termination_signal
from check_env import check_runtime, EXPECTED_VERSIONS
from baseline import prepare_model, run as run_baseline


class CoreTests(unittest.TestCase):
    def test_prepared_snapshot_and_local_directory_share_the_same_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"fake fixture")
            download = Mock(return_value=str(snapshot))
            with patch.dict(sys.modules, {
                    "huggingface_hub": SimpleNamespace(snapshot_download=download),
                    "huggingface_hub.utils": SimpleNamespace(enable_progress_bars=lambda: None),
            }), redirect_stdout(io.StringIO()):
                ready = prepare_model("Qwen/test-model", revision="fixed-model-commit")
                local_ready = prepare_model(ready)
            self.assertEqual(Path(ready).resolve(), snapshot.resolve())
            self.assertEqual(local_ready, str(snapshot.resolve()))
            self.assertEqual(download.call_count, 1)
            self.assertEqual(download.call_args.kwargs["revision"], "fixed-model-commit")

    def test_incomplete_shards_fail_before_gpu_engine_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "part-1.safetensors").write_bytes(b"fake fixture")
            (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                "layer1": "part-1.safetensors", "layer2": "part-2.safetensors",
            }}), encoding="utf-8")
            output = root / "output"
            output.mkdir()
            (output / "model_configs.json").write_text(json.dumps({"Qwen/test-model": {
                "resolved_revision": "pinned-commit",
            }}), encoding="utf-8")
            args = SimpleNamespace(model="Qwen/test-model", output=output)
            engine = Mock()
            download = Mock(return_value=str(snapshot))
            with patch.dict(sys.modules, {
                    "huggingface_hub": SimpleNamespace(snapshot_download=download),
                    "huggingface_hub.utils": SimpleNamespace(enable_progress_bars=lambda: None),
                    "vllm": SimpleNamespace(LLM=engine, SamplingParams=Mock()),
            }), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "part-2.safetensors"):
                    run_baseline(args)
            engine.assert_not_called()
            self.assertEqual(download.call_args.kwargs["revision"], "pinned-commit")
            self.assertFalse((output / "SUCCESS").exists())

    def test_uuid_selection_is_rejected_before_cuda_initialization(self):
        # PyTorch 接受 UUID，但固定版本 vLLM 随后会在 int(UUID) 处失败。
        # 检查必须在初始化 CUDA/分配显存之前拦截该配置。
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-37567038-7965-d679-03b2-977a323a5456"}), \
                patch("check_env.platform.system", return_value="Linux"), \
                patch("check_env.importlib.metadata.version", side_effect=EXPECTED_VERSIONS.__getitem__), \
                patch.dict(sys.modules, {"torch": SimpleNamespace()}), \
                redirect_stdout(io.StringIO()):
            errors = check_runtime(1)
        self.assertTrue(any("不兼容 UUID" in error and "CUDA_VISIBLE_DEVICES=7" in error for error in errors))

    def test_signal_exit_is_distinct_from_python_exception(self):
        self.assertEqual(termination_signal(-15), "SIGTERM")
        self.assertIsNone(termination_signal(1))
        self.assertIsNone(termination_signal(0))
        self.assertEqual(termination_signal(-999), "signal 999")

    def test_failure_log_excerpt_is_bounded_and_handles_empty_log(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run.log"
            path.write_text("", encoding="utf-8")
            self.assertEqual(log_tail(path), "")
            # 大日志开头不应输出；末尾的中文与最后一条错误应完整保留。
            path.write_text("old line\n" * 10000 + "模型加载中\nlast error\n", encoding="utf-8")
            excerpt = log_tail(path)
            self.assertEqual(len(excerpt.splitlines()), 80)
            self.assertTrue(excerpt.endswith("模型加载中\nlast error"))
            self.assertLessEqual(len(excerpt.encode("utf-8")), 65536)

    def test_failed_child_preserves_evidence_and_has_no_success_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "failed"
            args = SimpleNamespace(draft_model="fake-small", target_model="fake-large", seed=0)
            config_loader = SimpleNamespace(from_pretrained=lambda name: SimpleNamespace(to_dict=lambda: {}))
            command = [sys.executable, "-u", "-c", "import sys; print('model load failed'); sys.exit(1)"]
            errors = io.StringIO()
            console = io.StringIO()
            with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoConfig=config_loader)}), \
                    patch.dict("os.environ", {"VLLM_ATTENTION_BACKEND": "XFORMERS", "CUDA_LAUNCH_BLOCKING": "1"}), \
                    patch("run.subprocess.run", return_value=SimpleNamespace(stdout="", stderr="")), \
                    redirect_stdout(console), \
                    redirect_stderr(errors):
                with self.assertRaisesRegex(RuntimeError, "退出码 1"):
                    execute(args, "small", directory, command)
            config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["exit_code"], 1)
            self.assertEqual(config["attention_backend_requested"], "XFORMERS")
            self.assertEqual(config["cuda_launch_blocking"], "1")
            self.assertIsNone(config["termination_signal"])
            self.assertGreater(config["child_pid"], 0)
            self.assertGreaterEqual(config["process_seconds"], 0)
            self.assertIn("finished_at_utc", config)
            self.assertFalse(config["interrupted_by_user"])
            self.assertIn("model load failed", console.getvalue())
            self.assertIn("model load failed", errors.getvalue())
            self.assertIn("model load failed", (directory / "run.log").read_text())
            self.assertFalse((directory / "SUCCESS").exists())

    def test_user_interrupt_stops_child_and_preserves_partial_log(self):
        class InterruptedOutput:
            def __iter__(self):
                yield "loading weights\n"
                raise KeyboardInterrupt

        class Child:
            pid = 12345
            stdout = InterruptedOutput()
            returncode = None
            terminated = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def communicate(self, timeout=None):
                return "Cancellation requested\n", None

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "interrupted"
            args = SimpleNamespace(draft_model="fake-small", target_model="fake-large", seed=0)
            config_loader = SimpleNamespace(from_pretrained=lambda name: SimpleNamespace(to_dict=lambda: {}))
            child = Child()
            console = io.StringIO()
            with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoConfig=config_loader)}), \
                    patch("run.subprocess.run", return_value=SimpleNamespace(stdout="", stderr="")), \
                    patch("run.subprocess.Popen", return_value=child), redirect_stdout(console):
                with self.assertRaises(KeyboardInterrupt):
                    execute(args, "large", directory, ["fake-model-command"])
            config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
            self.assertTrue(child.terminated)
            self.assertTrue(config["interrupted_by_user"])
            self.assertEqual(config["exit_code"], -15)
            self.assertEqual(config["termination_signal"], "SIGTERM")
            self.assertIn("finished_at_utc", config)
            log = (directory / "run.log").read_text(encoding="utf-8")
            self.assertEqual(log, "loading weights\nCancellation requested\n")
            self.assertIn(log, console.getvalue())
            self.assertFalse((directory / "SUCCESS").exists())

    def test_discarded_tokens_are_charged(self):
        # 两题，共 150 个小模型计费 token、200 个大模型 token。
        # 费用应为 (0.45 + 2.8)/2 = 1.625 TFLOPs/题，不能漏掉首步丢弃。
        samples = [
            {"idx": 0, "score": [True], "pred": ["2"], "token_counts": [[100, 100, 50]]},
            {"idx": 1, "score": [False], "pred": [None], "token_counts": [[0, 100, 0]]},
        ]
        result = calculate(samples)
        self.assertAlmostEqual(result["mean_flops_t"], 1.625)
        self.assertEqual(result["rule_accuracy"], 50)
        self.assertEqual(result["empty_answers"], 1)

    def test_invalid_or_duplicate_results_are_rejected(self):
        sample = {"idx": 0, "score": [True], "pred": ["1"], "token_counts": [[1, 0, 0]]}
        with self.assertRaises(ValueError):
            calculate([sample, sample])
        with self.assertRaises(ValueError):
            calculate([{**sample, "token_counts": [[-1, 0, 0]]}])
        with self.assertRaises(ValueError):
            calculate([{**sample, "score": []}])

    def test_only_completed_comparable_runs_get_savings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            common = {"samples": 1, "seed": 0, "draft_model": "small", "target_model": "large",
                      "temperature": 0.7, "top_p": 1, "max_tokens": 4096,
                      "max_model_len": 8192, "small_params": 1.5e9, "large_params": 7e9,
                      "dataset_sha256": "same"}
            for method, tokens in [("large", [0, 100, 0]), ("steer", [100, 50, 50]), ("small", [100, 0, 0])]:
                directory = root / method
                directory.mkdir()
                (directory / "run_config.json").write_text(json.dumps({**common, "method": method}), encoding="utf-8")
                sample = {"idx": 0, "score": [True], "pred": ["1"], "token_counts": [tokens]}
                (directory / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
                # 未完成 small 不可被统计；未评分文件也不应重复计入。
                (directory / "samples_unscored.jsonl").write_text(json.dumps(sample), encoding="utf-8")
                if method != "small":
                    (directory / "SUCCESS").touch()
            with redirect_stdout(io.StringIO()):
                rows = summarize(root)
            self.assertEqual({row["method"] for row in rows}, {"large", "steer"})
            steer = next(row for row in rows if row["method"] == "steer")
            self.assertAlmostEqual(steer["saving_vs_large_pct"], 100 * (1 - 1.15 / 1.4))

    def test_patch_is_repeatable_and_restores_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            originals = {}
            for rel in ["sequence.py", "model_executor/layers/sampler.py"]:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                originals[rel] = ("original:" + rel).encode()
                path.write_bytes(originals[rel])
            with redirect_stdout(io.StringIO()):
                apply_patch(root)
                apply_patch(root)
                apply_patch(root, restore=True)
            for rel, content in originals.items():
                self.assertEqual((root / rel).read_bytes(), content)
                self.assertEqual((root / rel).with_name(Path(rel).name + ".steer.bak").read_bytes(), content)

    def test_dry_run_needs_no_gpu_and_writes_no_results(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "未创建的结果"
            result = subprocess.run([sys.executable, str(ROOT / "run.py"), "--dry-run",
                                     "--samples", "50", "--output", str(output)],
                                    capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(b"--reliability_k_top 1", result.stdout)
            self.assertFalse(output.exists())

    def test_all_local_imports_exist_after_trimming(self):
        # compileall 无法发现缺失模块；检查实际代码里的项目内 import。
        for path in ROOT.rglob("*.py"):
            if "vllm_overwrite" in path.parts:
                continue  # 这些补丁文件的 import 在安装后的 vLLM 包内解析。
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                if node.level:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    target = base.joinpath(*node.module.split("."))
                elif node.module.startswith("external."):
                    target = ROOT.joinpath(*node.module.split("."))
                else:
                    continue
                self.assertTrue(target.with_suffix(".py").exists() or (target / "__init__.py").exists(),
                                f"缺失本地模块: {path.name}: {node.module}")

    def test_official_loop_replaces_first_step_and_charges_discard(self):
        # 执行真实 get_responses，只有 GPU/路由判定被替换为确定的测试输入。
        # 两道题中第二道的首步被大模型替换，下一步两道均由小模型完成。
        source = (ROOT / "ours_main.py").read_text(encoding="utf-8")
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == "get_responses")

        class Tensor(list):
            def dim(self):
                return 1
            def float(self):
                return self

        def completion(text, count):
            logs = [{0: SimpleNamespace(decoded_token="2", top_logits_list=[10.])}
                    for _ in range(count)]
            return SimpleNamespace(outputs=[SimpleNamespace(text=text, logprobs=logs,
                                                            token_ids=list(range(count)), stop_reason="\n\n")])

        draft_batches = [[completion("Small step\n\n", 2), completion("Wrong step\n\n", 3)],
                         [completion(r"\boxed{2}" + "\n\n", 4), completion(r"\boxed{3}" + "\n\n", 4)]]
        target_batches = [[completion("Fixed step\n\n", 2)]]

        def client(batches):
            def generate(**kwargs):
                batch = batches.pop(0)
                self.assertEqual(len(batch), len(kwargs["prompts"]))
                return batch
            return SimpleNamespace(generate=SimpleNamespace(remote=generate))

        namespace = {"torch": SimpleNamespace(tensor=Tensor),
                     "ray": SimpleNamespace(get=lambda value: value),
                     "SamplingParams": lambda **kwargs: SimpleNamespace(**kwargs),
                     "dirichlet_au_eu": lambda *args, **kwargs: {"eu": 0.1},
                     "get_step_reliability": lambda **kwargs: 10.,
                     "fit_mixture_model": lambda values: (None, None, None, None),
                     "get_gmm_responsibility": lambda *args, **kwargs: ([0], [1], []),
                     "route_prompts": lambda *args: ([0, 1], [], [])}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "official_get_responses", "exec"), namespace)
        args = SimpleNamespace(step_word="\n\n", max_tokens_per_call=4096, max_model_len=8192,
                               temperature=.7, top_p=1, seed=0, reliability_metric="R_eu",
                               reliability_k_top=1, reliability_mode="math_only_avg",
                               reliability_target_usage="gmm_responsibility", draft_gmm_threshold=.4,
                               target_gmm_threshold=.4, max_steps=100, patience=5)
        tokenizer = SimpleNamespace(encode=lambda text: list(text))
        with redirect_stdout(io.StringIO()):
            outputs, counts, turns, _, _, _ = namespace["get_responses"](
                args, client(draft_batches), client(target_batches), None,
                tokenizer, tokenizer, None, ["question 0", "question 1"], ["0", "1"])
        self.assertNotIn("Wrong step", outputs[1])
        self.assertIn("Fixed step", outputs[1])
        self.assertEqual(counts, [(6, 0, 0), (4, 2, 3)])
        self.assertEqual(turns[1], [(0, 2), (1, 1)])
        self.assertEqual(draft_batches + target_batches, [])


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ["numpy", "sklearn", "sympy", "antlr4", "regex", "pebble", "word2number"]),
                     "需要安装 CPU 评分/GMM 依赖；此测试不会下载模型")
class MathTests(unittest.TestCase):
    def test_real_math_parser_and_grader(self):
        from external.qwen25_math_evaluation.parser import extract_answer, parse_ground_truth
        from external.qwen25_math_evaluation.grader import math_equal
        self.assertEqual(extract_answer(r"Final: \boxed{\frac{1}{2}}", "math500"), r"\frac{1}{2}")
        self.assertTrue(math_equal("0.5", r"\frac{1}{2}"))
        self.assertFalse(math_equal("2", "3"))
        _, answer = parse_ground_truth({"solution": r"We obtain \boxed{7}.", "answer": "7"})
        self.assertEqual(answer, "7")

    def test_gmm_handles_empty_math_and_small_groups(self):
        import numpy as np
        from utils import fit_mixture_model, get_gmm_responsibility, route_prompts
        values = [(i, -np.inf) for i in range(10)]
        self.assertEqual(get_gmm_responsibility(values, None), ([], list(range(10)), []))
        _, _, _, model = fit_mixture_model([-np.inf] * 10)
        self.assertIsNone(model)
        self.assertEqual(route_prompts(None, [(0, 10.)], [], .4, .4), ([], [0], []))


if __name__ == "__main__":
    unittest.main()
