import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from build_sft_data import (CYCLIC_ORDERS, LETTERS4, REMAINING_ORDERS,
                            load_jsonl, permute_record_order)


class ChineseRemainingPermutationTests(unittest.TestCase):
    def test_twenty_orders_are_unique_and_non_cyclic(self):
        self.assertEqual(len(REMAINING_ORDERS), 20)
        self.assertEqual(len(set(REMAINING_ORDERS)), 20)
        self.assertFalse(set(REMAINING_ORDERS).intersection(CYCLIC_ORDERS))

    def test_gold_is_remapped_to_its_new_position(self):
        record = {
            "options": dict(zip(LETTERS4, ["one", "two", "three", "four"])),
            "gold": "B",
        }
        for order in REMAINING_ORDERS:
            permuted, old_to_new = permute_record_order(record, order)
            self.assertEqual(permuted["gold"], old_to_new["B"])
            self.assertEqual(
                permuted["options"][permuted["gold"]],
                record["options"]["B"],
            )

    def test_generated_file_has_twenty_rows_per_uid(self):
        rows = load_jsonl(
            ROOT / "sft_data" /
            "zh_train_remaining_permutations.jsonl")
        self.assertEqual(len(rows), 15800)
        by_uid = defaultdict(list)
        for row in rows:
            meta = row["meta"]
            self.assertEqual(
                meta["augmentation_source"], "remaining_permutation")
            by_uid[meta["uid"]].append(meta["permutation_order"])
        self.assertEqual(len(by_uid), 790)
        self.assertEqual(
            Counter(len(orders) for orders in by_uid.values()),
            Counter({20: 790}),
        )
        self.assertTrue(all(len(set(orders)) == 20
                            for orders in by_uid.values()))


if __name__ == "__main__":
    unittest.main()
