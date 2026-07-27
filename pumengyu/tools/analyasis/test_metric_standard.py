from __future__ import annotations

import unittest

from pumengyu.tools.analyasis.metric_standard import aggregate_liver_tumor_metrics


def _positive(dice: float, liver: float = 0.9) -> dict:
    return {
        "dice": dice,
        "jaccard": dice / (2 - dice),
        "recall": dice,
        "fnr": 1 - dice,
        "precision": dice,
        "fdr": 1 - dice,
        "liver_dice": liver,
    }


class MetricStandardTests(unittest.TestCase):
    def test_negative_fp_does_not_change_primary_tumor_dice(self) -> None:
        positives = [_positive(0.8), _positive(0.6)]
        negatives = [
            {"liver_dice": 0.95, "pred_tumor": 0},
            {"liver_dice": 0.85, "pred_tumor": 10},
        ]
        result = aggregate_liver_tumor_metrics(positives, negatives)
        self.assertAlmostEqual(result["primary"]["dice"]["mean"], 0.7)
        self.assertAlmostEqual(result["primary"]["precision"]["mean"], 0.7)
        self.assertEqual(result["n_negative_tn"], 1)
        self.assertEqual(result["n_negative_fp"], 1)
        self.assertAlmostEqual(result["negative_fp_rate"], 0.5)
        self.assertAlmostEqual(result["nnunet_reference"]["tumor_dice"]["mean"], 1.4 / 3)

    def test_overall_uses_all_case_liver_and_positive_only_tumor(self) -> None:
        positives = [_positive(0.8, liver=0.9), _positive(0.6, liver=0.8)]
        negatives = [
            {"liver_dice": 1.0, "pred_tumor": 0},
            {"liver_dice": 0.7, "pred_tumor": 100},
        ]
        result = aggregate_liver_tumor_metrics(positives, negatives)
        self.assertAlmostEqual(result["liver"]["mean"], 0.85)
        self.assertAlmostEqual(result["primary"]["dice"]["mean"], 0.7)
        self.assertAlmostEqual(result["primary"]["overall"], 0.775)

    def test_all_positive_dataset_matches_nnunet_reference(self) -> None:
        result = aggregate_liver_tumor_metrics([_positive(0.8), _positive(0.6)], [])
        self.assertAlmostEqual(
            result["primary"]["dice"]["mean"],
            result["nnunet_reference"]["tumor_dice"]["mean"],
        )
        self.assertAlmostEqual(
            result["primary"]["overall"], result["nnunet_reference"]["overall"]
        )


if __name__ == "__main__":
    unittest.main()
