"""四状态实验：配对方向、条件分母、固定子集与端到端结果保护。"""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import experiment1_four_states as experiment
from experiments.preview_experiment1_archive import ARCHIVE, reconstruct


class FourStateTests(unittest.TestCase):
    def test_archive_reconstruction_matches_all_marginals_and_detects_bad_inputs(self):
        self.assertEqual(reconstruct(ARCHIVE[0]), {"11": 355, "01": 59, "10": 20, "00": 66})
        counts = {state: sum(reconstruct(record)[state] for record in ARCHIVE) for state in experiment.STATES}
        self.assertEqual(counts, {"11": 1402, "01": 228, "10": 85, "00": 285})
        with self.assertRaises(ValueError):
            reconstruct({**ARCHIVE[0], "small_correct": 374})

    def test_state_direction_and_conditional_denominators(self):
        rows = [{"state": state} for state in ["11"] * 4 + ["01"] * 3 + ["10"] + ["00"] * 2]
        result = experiment.distribution(rows)
        self.assertEqual(result["counts"], {"11": 4, "01": 3, "10": 1, "00": 2})
        self.assertEqual(result["small_accuracy_pct"], 50)
        self.assertEqual(result["large_accuracy_pct"], 70)
        self.assertEqual(result["large_minus_small_pp"], 20)
        self.assertEqual(result["large_rescue_given_small_wrong_pct"], 60)
        self.assertEqual(result["large_wrong_given_small_correct_pct"], 20)
        self.assertEqual(result["answer_selection_oracle_accuracy_pct"], 80)
        self.assertIsNone(experiment.distribution([{"state": "11"}])["large_rescue_given_small_wrong_pct"])
        self.assertIsNone(experiment.distribution([{"state": "00"}])["large_wrong_given_small_correct_pct"])

    def test_pair_by_id_not_file_order_and_reject_mismatched_generation(self):
        dataset = {idx: {"unique_id": str(idx)} for idx in range(4)}
        common = dict(temperature=.7, top_p=1, max_tokens=4096, max_model_len=8192)
        def sample(correct):
            return {"score": [correct], "pred": ["answer"]}
        small = {"config": dict(common), "samples": {3: sample(0), 2: sample(1), 1: sample(0), 0: sample(1)}}
        large = {"config": dict(common), "samples": {0: sample(1), 1: sample(1), 2: sample(0), 3: sample(0)}}
        rows = experiment.pair_runs(small, large, dataset, list(range(4)), 0)
        self.assertEqual([row["state"] for row in rows], ["11", "01", "10", "00"])
        large["config"]["max_tokens"] = 2048
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            experiment.pair_runs(small, large, dataset, list(range(4)), 0)

    def test_fixed_subset_and_cross_seed_stability(self):
        dataset = dict.fromkeys(range(500))
        ids = experiment.select_ids(dataset, 300, 12)
        self.assertEqual(ids, experiment.select_ids(dataset, 300, 12))
        self.assertEqual(len(set(ids)), 300)
        self.assertEqual(ids, sorted(ids))
        self.assertNotEqual(ids, experiment.select_ids(dataset, 300, 13))
        rows = [{"idx": 0, "unique_id": "q0", "level": 1, "subject": "Algebra", "seed": seed,
                 "state": state} for seed, state in ((0, "01"), (1, "00"))]
        per_seed, _, stability, _ = experiment.state_tables(rows, [0], [0, 1])
        self.assertEqual(len(per_seed), 2)
        self.assertFalse(stability[0]["same_state_all_seeds"])
        self.assertFalse(stability[0]["both_wrong_all_seeds"])

    def fixture(self, root, seeds=(0, 1)):
        data_path = root / "data/math500/test.jsonl"
        data_path.parent.mkdir(parents=True)
        dataset = [{"problem": f"problem {idx}", "answer": "2", "unique_id": f"test/{idx}",
                    "level": idx % 5 + 1, "subject": "Algebra"} for idx in range(500)]
        data_path.write_text("".join(json.dumps(row) + "\n" for row in dataset), encoding="utf-8")
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        (root / "UPSTREAM.json").write_text(json.dumps({"math500_sha256": digest}), encoding="utf-8")
        args = SimpleNamespace(small_root=root / "small", large_root=root / "large", seeds=list(seeds),
                               samples=-1, subset_seed=0, output=root / "report", no_plots=True,
                               plot_language="en", font_path=None)
        for method in ("small", "large"):
            for seed in seeds:
                directory = getattr(args, f"{method}_root") / f"seed_{seed}" / method
                directory.mkdir(parents=True)
                config = {"method": method, "seed": seed, "samples": -1, "dataset_sha256": digest,
                          "draft_model": "small", "target_model": "large", "temperature": .7,
                          "top_p": 1, "max_tokens": 4096, "max_model_len": 8192}
                (directory / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
                rows = []
                for idx, canonical in enumerate(dataset):
                    state = experiment.STATES[(idx + seed) % 4]
                    correct = int(state[0 if method == "small" else 1])
                    rows.append({**canonical, "idx": idx, "score": [correct], "pred": ["2"],
                                 "code": ["complete answer"],
                                 "token_counts": [[1, 0, 0] if method == "small" else [0, 1, 0]]})
                (directory / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reversed(rows)), encoding="utf-8")
                (directory / "metrics.json").write_text('{"acc":50.0,"num_samples":500}', encoding="utf-8")
                (directory / "SUCCESS").write_text("validated", encoding="utf-8")
        return args

    def test_full_analysis_keeps_inputs_and_distinguishes_questions_from_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.fixture(root)
            originals = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with patch.object(experiment, "ROOT", root), redirect_stdout(io.StringIO()):
                summary = experiment.analyze(args)
            self.assertEqual(summary["num_unique_questions"], 500)
            self.assertEqual(summary["pooled"]["num_pairs"], 1000)
            self.assertEqual(summary["pooled"]["counts"], dict.fromkeys(experiment.STATES, 250))
            self.assertTrue((args.output / "paired_answers.jsonl").is_file())
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)
            with patch.object(experiment, "ROOT", root), self.assertRaises(FileExistsError):
                experiment.analyze(args)

    def test_changed_question_failed_run_and_different_seed_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.fixture(root)
            path = args.large_root / "seed_0/large/samples.jsonl"
            original = path.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in original.splitlines()]
            rows[0]["problem"] = "changed"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            with patch.object(experiment, "ROOT", root), self.assertRaisesRegex(ValueError, "problem"):
                experiment.analyze(args)
            path.write_text(original, encoding="utf-8")
            success = args.small_root / "seed_0/small/SUCCESS"
            success.unlink()
            with patch.object(experiment, "ROOT", root), self.assertRaisesRegex(ValueError, "成功"):
                experiment.analyze(args)
            success.write_text("validated", encoding="utf-8")
            path = args.large_root / "seed_1/large/run_config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["temperature"] = .6
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(experiment, "ROOT", root), self.assertRaisesRegex(ValueError, "temperature"):
                experiment.analyze(args)
            self.assertFalse(args.output.exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "绘图依赖未安装")
    def test_real_plot_export_for_single_seed_and_fixed_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.fixture(root, seeds=(0,))
            args.samples = 300
            args.no_plots = False
            with patch.object(experiment, "ROOT", root), redirect_stdout(io.StringIO()):
                summary = experiment.analyze(args)
            self.assertEqual(summary["pooled"]["num_pairs"], 300)
            for name in ("four_states", "four_states_by_group"):
                for ext in ("png", "pdf", "svg"):
                    self.assertGreater((args.output / f"{name}.{ext}").stat().st_size, 500)
            self.assertIsNone(summary["across_seed_state_percentages"]["01"]["sample_sd_pp"])


if __name__ == "__main__":
    unittest.main()
