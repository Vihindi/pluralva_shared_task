import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mixed_microbatching import (build_mixed_optimizer_blocks,
                                 load_schedule_rows,
                                 mixed_schedule_stats)


class MixedMicrobatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_schedule_rows([
            ROOT / "sft_data" / "zh_train_full.jsonl",
            ROOT / "sft_data" / "id_train_full.jsonl",
            ROOT / "sft_data" / "si_train_full.jsonl",
            ROOT / "sft_data" / "zh_train_remaining_permutations.jsonl",
        ])
        cls.blocks = build_mixed_optimizer_blocks(cls.rows, seed=42)

    def test_epoch_distribution(self):
        self.assertEqual(len(self.blocks), 814)
        stats = mixed_schedule_stats(self.rows, self.blocks)
        self.assertEqual(
            stats,
            {
                "optimizer_blocks": 814,
                "rows": {
                    "indonesian": 7326,
                    "chinese": 5698,
                    "sri_lankan": 3256,
                },
                "chinese_sources": {
                    "primary": 3160,
                    "remaining_permutation": 2538,
                },
            },
        )

    def test_every_block_is_nine_seven_four_with_unique_id_and_zh(self):
        expected = Counter({
            "indonesian": 9,
            "chinese": 7,
            "sri_lankan": 4,
        })
        for block in self.blocks:
            self.assertEqual(len(block), 20)
            self.assertEqual(
                Counter(self.rows[index]["dataset"] for index in block),
                expected,
            )
            for dataset in ("indonesian", "chinese"):
                uids = [
                    self.rows[index]["uid"] for index in block
                    if self.rows[index]["dataset"] == dataset
                ]
                self.assertEqual(len(uids), len(set(uids)))

    def test_primary_chinese_is_exhausted_before_fallback(self):
        extra_by_step = []
        for block in self.blocks:
            extra_by_step.append(sum(
                self.rows[index]["augmentation_source"] ==
                "remaining_permutation"
                for index in block
                if self.rows[index]["dataset"] == "chinese"
            ))
        self.assertEqual(extra_by_step[:451], [0] * 451)
        self.assertEqual(extra_by_step[451], 4)
        self.assertEqual(extra_by_step[452:], [7] * (814 - 452))

    def test_coverage_first_cycles(self):
        uses = Counter(index for block in self.blocks for index in block)
        by_dataset = defaultdict(list)
        for index, row in enumerate(self.rows):
            by_dataset[row["dataset"]].append(index)

        primary_zh = [
            index for index in by_dataset["chinese"]
            if self.rows[index]["augmentation_source"] == "primary"
        ]
        extra_zh = [
            index for index in by_dataset["chinese"]
            if self.rows[index]["augmentation_source"] ==
            "remaining_permutation"
        ]
        self.assertEqual({uses[index] for index in primary_zh}, {1})
        self.assertEqual(sum(bool(uses[index]) for index in extra_zh), 2538)
        self.assertEqual({uses[index] for index in extra_zh}, {0, 1})

        id_uses = [uses[index] for index in by_dataset["indonesian"]]
        self.assertEqual(Counter(id_uses), Counter({1: 7314, 2: 6}))

        si_uses = [uses[index] for index in by_dataset["sri_lankan"]]
        self.assertEqual(Counter(si_uses), Counter({5: 200, 6: 376}))


if __name__ == "__main__":
    unittest.main()

