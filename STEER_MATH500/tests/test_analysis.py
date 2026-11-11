"""已有结果分析的离线检查：不下载、不启动 GPU、不需要数学评分依赖。"""
from copy import deepcopy
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import analyze_results as analysis


def canonical(idx):
    return {"idx": idx, "problem": f"problem {idx}", "answer": "2", "solution": "2",
            "unique_id": f"test/{idx}", "level": idx % 5 + 1, "subject": "Algebra"}


def sample(idx, method="small"):
    row = {**canonical(idx), "gt": "2", "code": ["answer: 2"], "pred": ["2"], "score": [True],
           "token_counts": [[10, 0, 0] if method == "small" else [0, 10, 0]]}
    if method == "steer":
        row.update({"token_counts": [[5, 5, 2]], "turn_info": [[[0, 1], [1, 2], [2, 1]]],
                    "step_reliabilities": [[{"step": i, "client_id": client}
                                             for i, client in row_turns()]]})
    return row


def row_turns():
    return [(0, 1), (1, 2), (2, 1)]


class AnalysisTests(unittest.TestCase):
    def test_missing_duplicate_or_changed_question_is_rejected(self):
        dataset = {idx: canonical(idx) for idx in range(3)}
        rows = [sample(idx) for idx in range(3)]
        for bad in (rows[:-1], rows + [deepcopy(rows[0])]):
            with self.assertRaises(ValueError):
                analysis.validate_samples(bad, dataset, "small")
        rows[1]["problem"] = "another question"
        with self.assertRaises(ValueError):
            analysis.validate_samples(rows, dataset, "small")

    def test_boolean_token_counts_and_nonbinary_scores_are_rejected(self):
        dataset = {0: canonical(0)}
        row = sample(0)
        row["token_counts"] = [[True, 0, 0]]
        with self.assertRaises(ValueError):
            analysis.validate_samples([row], dataset, "small")
        row = sample(0)
        row["score"] = [0.5]
        with self.assertRaises(ValueError):
            analysis.validate_samples([row], dataset, "small")

    def test_routing_and_discarded_token_cost_without_inventing_stop_reason(self):
        row = sample(0, "steer")
        config = {"small_params": 1.5e9, "large_params": 7e9, "max_steps": 3}
        stats = analysis.question_stats(row, config, "steer")
        self.assertAlmostEqual(stats["flops_t"], (3e9 * 7 + 14e9 * 5) / 1e12)
        self.assertEqual(stats["small_steps"], 2)
        self.assertEqual(stats["large_steps"], 1)
        self.assertEqual(stats["switches"], 2)
        self.assertTrue(stats["max_steps_boundary"])
        self.assertEqual(stats["finish_reason"], "not_recorded")
        self.assertFalse(stats["patience_flag_recorded"])

    def test_regrade_reextracts_and_matches_ids_without_mutating_original(self):
        dataset = {idx: canonical(idx) for idx in range(2)}
        old = {idx: sample(idx) for idx in range(2)}
        before = deepcopy(old)
        run = {"samples": old, "method": "small", "seed": 0}
        def extractor(executor, code, prompt, data):
            self.assertEqual(code, "answer: 2")
            return "fresh 2", None
        def evaluator(data, prompt, samples):
            for item in samples:
                item["score"] = [item["idx"] != 1]
                item["gt"] = "2"
            return list(reversed(samples)), {"acc": 50.0, "timeout_samples": 0}
        changes = analysis.regrade_run(run, dataset, evaluator, extractor)
        self.assertEqual(old, before)
        self.assertEqual(len(changes), 2)
        self.assertTrue(changes[0]["prediction_changed"])
        self.assertFalse(changes[0]["score_changed"])
        self.assertTrue(changes[1]["score_changed"])

    def test_regrade_failure_or_missing_ids_does_not_become_success(self):
        dataset = {0: canonical(0)}
        run = {"samples": {0: sample(0)}, "method": "small", "seed": 0}
        extract = lambda *args: ("2", None)
        def exits(*args, **kwargs):
            raise SystemExit(0)
        with self.assertRaises(RuntimeError):
            analysis.regrade_run(run, dataset, exits, extract)
        with self.assertRaises(ValueError):
            analysis.regrade_run(run, dataset, lambda *args, **kwargs: ([], {}), extract)

    def test_paired_gains_and_losses_are_not_conflated(self):
        rows = [{"small_score": 0, "large_score": 1, "steer_score": 1},
                {"small_score": 1, "large_score": 1, "steer_score": 0},
                {"small_score": 0, "large_score": 0, "steer_score": 0}]
        result = analysis.paired_counts(rows)
        self.assertEqual(result["steer_rescues_small"], 1)
        self.assertEqual(result["steer_loses_small"], 1)
        self.assertEqual(result["steer_loses_large"], 1)
        self.assertEqual(result["all_wrong"], 1)

    def prepare_runs(self, root):
        data = root / "data/math500/test.jsonl"
        data.parent.mkdir(parents=True)
        rows = [canonical(idx) for idx in range(500)]
        data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        digest = hashlib.sha256(data.read_bytes()).hexdigest()
        args = SimpleNamespace(seeds=[0], regrade=False, output=root / "report")
        for method in analysis.METHODS:
            run_root = root / method
            setattr(args, f"{method}_root", run_root)
            directory = run_root / "seed_0" / method
            directory.mkdir(parents=True)
            config = {"method": method, "seed": 0, "samples": -1, "dataset_sha256": digest,
                      "small_params": 1.5e9, "large_params": 7e9, "max_steps": 100,
                      "draft_model": "small-model", "target_model": "large-model",
                      "temperature": 0.7, "top_p": 1.0, "max_tokens": 4096, "max_model_len": 8192}
            (directory / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            (directory / "samples.jsonl").write_text("".join(json.dumps(sample(i, method)) + "\n"
                                                             for i in range(500)), encoding="utf-8")
            (directory / "metrics.json").write_text('{"acc":100.0,"num_samples":500}', encoding="utf-8")
            (directory / "SUCCESS").write_text("validated", encoding="utf-8")
        return args, {row["idx"]: row for row in rows}, digest

    def test_full_analysis_writes_separate_reports_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, _, _ = self.prepare_runs(root)
            inputs = {path: path.read_bytes() for path in root.rglob("*.jsonl")}
            with patch.object(analysis, "ROOT", root), redirect_stdout(io.StringIO()):
                analysis.analyze(args)
            self.assertTrue((args.output / "report.md").is_file())
            self.assertTrue((args.output / "per_question.csv").is_file())
            summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["steer"]["accuracy_pct"]["mean"], 100.0)
            self.assertIsNone(summary["steer"]["accuracy_pct"]["sample_sd"])
            for path, original in inputs.items():
                self.assertEqual(path.read_bytes(), original)
            with patch.object(analysis, "ROOT", root), self.assertRaises(FileExistsError):
                analysis.analyze(args)

    def test_wrong_metrics_or_xformers_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, dataset, digest = self.prepare_runs(root)
            directory = args.small_root / "seed_0/small"
            (directory / "metrics.json").write_text('{"acc":90.0,"num_samples":500}', encoding="utf-8")
            with self.assertRaises(ValueError):
                analysis.audit_run(args.small_root, 0, "small", dataset, digest)
            directory = args.steer_root / "seed_0/steer"
            config_path = directory / "run_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["attention_backend_requested"] = "XFORMERS"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                analysis.audit_run(args.steer_root, 0, "steer", dataset, digest)


if __name__ == "__main__":
    unittest.main()
