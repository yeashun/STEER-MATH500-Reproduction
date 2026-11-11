"""实验 2 的排序统计、题目聚类、特征无标签依赖与状态分类检查。"""
from copy import deepcopy
import unittest

import experiment2_upgrade_confidence as experiment


class ConfidenceTests(unittest.TestCase):
    def test_auc_has_fixed_direction_and_half_credit_for_ties(self):
        labels = [0, 0, 1, 1]
        scores = [0, .5, .5, 1]
        self.assertEqual(experiment.auc(labels, scores), .875)
        self.assertEqual(experiment.auc(labels, [-s for s in scores]), .125)
        self.assertEqual(experiment.auc(labels, [1, 1, 1, 1]), .5)
        self.assertIsNone(experiment.auc([1, 1], [.1, .9]))
        with self.assertRaises(ValueError):
            experiment.auc(labels, [0, 1])
        with self.assertRaises(ValueError):
            experiment.auc([0, 1], [0, float("nan")])

    def test_agreement_excludes_self_and_never_reads_scores_or_ground_truth(self):
        source = {seed: {0: {"pred": [prediction], "score": [True], "gt": "unused"}}
                  for seed, prediction in enumerate((" 2 ", "2", "3", "4"))}
        result = experiment.answer_agreement(source, [0])
        self.assertEqual(result[0, 0]["answer_agreement_proxy"], 1 / 3)
        self.assertEqual(result[2, 0]["answer_agreement_proxy"], 0)
        changed = deepcopy(source)
        for rows in changed.values():
            rows[0]["score"] = [False]
            rows[0]["gt"] = "different gold answer"
        self.assertEqual(experiment.answer_agreement(changed, [0]), result)
        # 改变当前答案，另外三个答案不会凭“自投票”凭空增加一票。
        changed[0][0]["pred"] = ["unique"]
        self.assertEqual(experiment.answer_agreement(changed, [0])[0, 0]["other_seed_matches"], 0)

    def test_empty_answers_are_missing_and_math_equivalence_is_not_assumed(self):
        source = {seed: {0: {"pred": [prediction]}}
                  for seed, prediction in enumerate(("", None, "0.5", r"\frac{1}{2}"))}
        result = experiment.answer_agreement(source, [0])
        self.assertIsNone(result[0, 0]["answer_agreement_proxy"])
        self.assertIsNone(result[1, 0]["answer_agreement_proxy"])
        self.assertEqual(result[2, 0]["answer_agreement_proxy"], 0)

    def test_bootstrap_keeps_all_observations_of_a_question_together(self):
        # 两个题簇各有四个 seed；若只抽到一个题簇，必须标为单类无效抽样。
        rows = [{"idx": idx, "state": state, "other_seed_matches": matches,
                 "answer_agreement_proxy": matches / 3}
                for idx, state, matches in ((0, "01", 3), (1, "00", 0)) for _ in range(4)]
        result = experiment.cluster_bootstrap(rows, [0, 1], 300, 9)
        self.assertEqual(result["higher_agreement_auc_ci95"], [1., 1.])
        self.assertEqual(result["lower_agreement_auc_ci95"], [0., 0.])
        self.assertGreater(result["valid_repetitions"], 0)
        self.assertLess(result["valid_repetitions"], 225)
        self.assertEqual(result, experiment.cluster_bootstrap(rows, [0, 1], 300, 9))

    def test_stable_common_failures_are_separate_from_seed_sensitive_failures(self):
        dataset = {idx: {"unique_id": str(idx), "problem": f"q{idx}"} for idx in range(4)}
        patterns = [("00",) * 4, ("01", "01", "00", "01"), ("11",) * 4, ("11", "01", "10", "00")]
        pairs = [{"idx": idx, "seed": seed, "state": state}
                 for idx, states in enumerate(patterns) for seed, state in enumerate(states)]
        rows, summary = experiment.question_patterns(pairs, dataset)
        self.assertEqual(summary["four_times_00"], 1)
        self.assertEqual(summary["one_00_three_01"], 1)
        self.assertEqual(summary["categories"]["mixed"], 2)
        self.assertEqual(rows[0]["large_rescue_rate_given_small_wrong"], 0)
        self.assertEqual(rows[1]["large_rescue_rate_given_small_wrong"], .75)
        self.assertIsNone(rows[2]["large_rescue_rate_given_small_wrong"])
        self.assertEqual(rows[3]["num_distinct_states"], 4)

    def test_conditional_population_and_missing_feature_coverage_are_explicit(self):
        rows = []
        for seed in range(4):
            for idx, state in ((0, "01"), (1, "00")):
                rows.append({"seed": seed, "idx": idx, "state": state,
                             "other_seed_matches": 0, "answer_agreement_proxy": 0})
        rows.append({"seed": 0, "idx": 2, "state": "00", "other_seed_matches": None,
                     "answer_agreement_proxy": None})
        result = experiment.summarize_proxy(rows, [0, 1, 2], 100, 0)
        self.assertEqual(result["selected_records"], 9)
        self.assertEqual(result["available_records"], 8)
        self.assertEqual(result["missing_records"], 1)
        self.assertEqual(result["class_counts_all"], {"01": 4, "00": 5})
        self.assertEqual(result["class_counts_available"], {"01": 4, "00": 4})
        self.assertEqual(result["auc_higher_agreement_predicts_01"], .5)
        self.assertEqual(result["discrete_distribution_overlap"], 1)


if __name__ == "__main__":
    unittest.main()
