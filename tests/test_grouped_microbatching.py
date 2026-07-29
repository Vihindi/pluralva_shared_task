import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grouped_microbatching import (build_grouped_optimizer_blocks,
                                   grouped_schedule_stats,
                                   load_schedule_rows)


class GroupedMicrobatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_schedule_rows([
            ROOT / "sft_data" / "zh_train_full.jsonl",
            ROOT / "sft_data" / "id_train_full.jsonl",
            ROOT / "sft_data" / "si_train_full.jsonl",
        ])
        cls.blocks = build_grouped_optimizer_blocks(
            cls.rows, optimizer_block_rows=20, seed=42)

    def test_full_data_distribution_and_step_counts(self):
        stats = grouped_schedule_stats(self.rows, self.blocks)
        self.assertEqual(
            stats,
            {
                "chinese": {
                    "rows": 3160,
                    "optimizer_blocks": 158,
                    "uids": 790,
                },
                "indonesian": {
                    "rows": 7320,
                    "optimizer_blocks": 366,
                    "uids": 366,
                },
                "sri_lankan": {
                    "rows": 576,
                    "optimizer_blocks": 29,
                    "uids": 288,
                },
            },
        )
        self.assertEqual(len(self.blocks), 553)

    def test_every_row_occurs_once_and_blocks_are_single_language(self):
        scheduled = []
        for block_number, block in enumerate(self.blocks):
            languages = {self.rows[index]["dataset"] for index in block}
            self.assertEqual(len(languages), 1)
            scheduled.extend(block)
        self.assertEqual(len(scheduled), len(self.rows))
        self.assertEqual(set(scheduled), set(range(len(self.rows))))
        self.assertTrue(all(count == 1
                            for count in Counter(scheduled).values()))

    def test_only_final_block_is_partial(self):
        self.assertTrue(all(len(block) == 20 for block in self.blocks[:-1]))
        self.assertEqual(len(self.blocks[-1]), 16)
        self.assertEqual(
            {self.rows[index]["dataset"] for index in self.blocks[-1]},
            {"sri_lankan"},
        )
        self.assertEqual(
            len({self.rows[index]["uid"] for index in self.blocks[-1]}),
            8,
        )

    def test_indonesian_physical_batches_hold_one_permutation(self):
        uid_blocks = defaultdict(set)
        for block in self.blocks:
            if self.rows[block[0]]["dataset"] != "indonesian":
                continue
            self.assertEqual(len(block), 20)
            uid_counts = Counter(self.rows[index]["uid"] for index in block)
            self.assertEqual(len(uid_counts), 4)
            self.assertEqual(set(uid_counts.values()), {5})
            prompts = []
            for start in range(0, 20, 5):
                microbatch = block[start:start + 5]
                microbatch_uids = {
                    self.rows[index]["uid"] for index in microbatch}
                self.assertEqual(len(microbatch_uids), 1)
                keys = {
                    tuple(
                        (message.get("role"), message.get("content"))
                        for message in self.rows[index]["messages"][:-1]
                    )
                    for index in microbatch
                }
                self.assertEqual(len(keys), 1)
                prompts.append(next(iter(keys)))
            for uid in uid_counts:
                uid_blocks[uid].add(id(block))
        self.assertEqual(len(uid_blocks), 366)
        self.assertEqual({len(blocks) for blocks in uid_blocks.values()}, {4})

    def test_chinese_steps_have_twenty_distinct_uids(self):
        uid_blocks = defaultdict(set)
        for block_number, block in enumerate(self.blocks):
            if self.rows[block[0]]["dataset"] != "chinese":
                continue
            uid_counts = Counter(self.rows[index]["uid"] for index in block)
            self.assertEqual(len(uid_counts), 20)
            self.assertEqual(set(uid_counts.values()), {1})
            for uid in uid_counts:
                uid_blocks[uid].add(block_number)
        self.assertEqual(len(uid_blocks), 790)
        self.assertEqual({len(blocks) for blocks in uid_blocks.values()}, {4})

    def test_sri_lankan_pairs_remain_together(self):
        uid_blocks = defaultdict(set)
        for block_number, block in enumerate(self.blocks):
            if self.rows[block[0]]["dataset"] != "sri_lankan":
                continue
            uid_counts = Counter(self.rows[index]["uid"] for index in block)
            self.assertIn(len(uid_counts), {8, 10})
            self.assertEqual(set(uid_counts.values()), {2})
            for uid in uid_counts:
                uid_blocks[uid].add(block_number)
        self.assertEqual(len(uid_blocks), 288)
        self.assertEqual({len(blocks) for blocks in uid_blocks.values()}, {1})


if __name__ == "__main__":
    unittest.main()
