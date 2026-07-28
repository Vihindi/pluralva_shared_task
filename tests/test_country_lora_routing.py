import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import make_submission
from country_lora_pipeline import load_and_validate_sft


class FakeScorer:
    instances = []

    def __init__(self, model, adapter=None, adapters=None, **kwargs):
        self.model = model
        self.adapter = adapter
        self.adapters = adapters
        self.active = None
        self.activations = []
        self.mcq_calls = []
        FakeScorer.instances.append(self)

    def set_adapter(self, name):
        self.active = name
        self.activations.append(name)


def fake_load_test_records(path, dataset):
    return [{"uid": f"{dataset}-1", "dataset": dataset}]


def fake_score_mcq(scorer, rec, n_perms, value_summaries):
    scorer.mcq_calls.append((rec["dataset"], scorer.active, n_perms))
    return {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}


def fake_score_si(scorer, rec, value_summaries):
    scorer.mcq_calls.append((rec["dataset"], scorer.active, None))
    return "A", 0.9, 0.1


def fake_score_si_4way(*args, **kwargs):
    raise AssertionError("country-specific routing must use binary SI scoring")


def fake_evaluate_module():
    module = types.ModuleType("evaluate")
    module.Scorer = FakeScorer
    module.load_test_records = fake_load_test_records
    module.score_mcq = fake_score_mcq
    module.score_si = fake_score_si
    module.score_si_4way = fake_score_si_4way
    return module


class SftValidationTests(unittest.TestCase):
    def test_rejects_non_binary_sri_lankan_target(self):
        row = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "Answer: Both"},
            ],
            "meta": {
                "uid": "SI-test",
                "dataset": "sri_lankan",
                "fold": 0,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "si_train_full.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Answer: Yes/No"):
                load_and_validate_sft(path, "sri_lankan")


class SubmissionRoutingTests(unittest.TestCase):
    def setUp(self):
        FakeScorer.instances.clear()
        self.original_evaluate = sys.modules.get("evaluate")
        sys.modules["evaluate"] = fake_evaluate_module()

    def tearDown(self):
        if self.original_evaluate is None:
            sys.modules.pop("evaluate", None)
        else:
            sys.modules["evaluate"] = self.original_evaluate

    @staticmethod
    def args(out_dir, country_specific):
        return SimpleNamespace(
            out_dir=str(out_dir),
            model="fake/base",
            adapter=None if country_specific else "runs/joint",
            country_specific=country_specific,
            zh_adapter="runs/zh" if country_specific else None,
            id_adapter="runs/id" if country_specific else None,
            si_adapter="runs/si" if country_specific else None,
            load_4bit=True,
            load_8bit=False,
            test_dir="unused",
            si_mode="binary",
            value_summaries="auto",
            n_perms=4,
        )

    def test_country_specific_switches_before_each_group_and_keeps_permutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            merged = make_submission.run_predict(
                self.args(Path(tmp), country_specific=True))

        scorer = FakeScorer.instances[-1]
        self.assertIsNone(scorer.adapter)
        self.assertEqual(
            scorer.adapters,
            {
                "chinese": "runs/zh",
                "indonesian": "runs/id",
                "sri_lankan": "runs/si",
            },
        )
        self.assertEqual(
            scorer.activations,
            ["chinese", "indonesian", "sri_lankan"],
        )
        self.assertEqual(
            scorer.mcq_calls,
            [
                ("chinese", "chinese", 4),
                ("indonesian", "indonesian", 4),
                ("sri_lankan", "sri_lankan", None),
            ],
        )
        self.assertEqual(set(merged), {
            "chinese-1", "indonesian-1", "sri_lankan-1"})

    def test_existing_single_adapter_path_does_not_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_submission.run_predict(
                self.args(Path(tmp), country_specific=False))

        scorer = FakeScorer.instances[-1]
        self.assertEqual(scorer.adapter, "runs/joint")
        self.assertIsNone(scorer.adapters)
        self.assertEqual(scorer.activations, [])
        self.assertEqual(
            scorer.mcq_calls[:2],
            [
                ("chinese", None, 4),
                ("indonesian", None, 4),
            ],
        )


if __name__ == "__main__":
    unittest.main()
