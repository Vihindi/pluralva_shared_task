import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from prompts import (SI_BIN_INSTR_NEGATIVE, SI_BIN_INSTR_NORMAL,
                     build_messages, is_si_negative_question)


def si_record(question):
    return {
        "uid": "SI-test",
        "dataset": "sri_lankan",
        "value_english": "Responsibility",
        "question": question,
        "options": {"A": "ප්‍රකාශය A", "B": "ප්‍රකාශය B"},
    }


class SinhalaBinaryPromptRoutingTests(unittest.TestCase):
    def test_normal_question_uses_normal_instruction(self):
        rec = si_record("නිවැරදි ක්‍රියාව කුමක්ද?")
        user = build_messages(
            rec, si_statement="A", value_summaries=None)[1]["content"]
        self.assertIn(SI_BIN_INSTR_NORMAL, user)
        self.assertNotIn(SI_BIN_INSTR_NEGATIVE, user)

    def test_negative_question_uses_negative_instruction_for_both_candidates(self):
        rec = si_record("මෙයින් නිවැරදි නොවන ප්‍රකාශය කුමක්ද?")
        self.assertTrue(is_si_negative_question(rec["question"]))
        for statement in ("A", "B"):
            user = build_messages(
                rec, si_statement=statement,
                value_summaries=None)[1]["content"]
            self.assertIn(SI_BIN_INSTR_NEGATIVE, user)

    def test_bare_no_prefix_is_not_a_router_rule(self):
        self.assertFalse(is_si_negative_question("නොහොත් වෙනත් පිළිතුර කුමක්ද?"))

    def test_correct_statement_does_not_match_wrong_statement(self):
        self.assertFalse(
            is_si_negative_question("නිවැරදි ප්‍රකාශය කුමක්ද?"))

    def test_scenario_negation_does_not_reverse_the_question(self):
        self.assertFalse(is_si_negative_question(
            "හානි සිදු නොවන ආකාරයෙන් ව්‍යාපාරයක් ලාභ ඉපයීමට කළ යුත්තේ කුමක්ද?"
        ))

    def test_negative_verb_question_routes_to_negative_instruction(self):
        self.assertTrue(is_si_negative_question(
            "ගුරුවරයෙකුගෙන් අපේක්ෂා නොකරන්නේ කුමක්ද?"
        ))

    def test_false_statement_routes_to_negative_instruction(self):
        self.assertTrue(is_si_negative_question(
            "පහත ප්‍රකාශවලින් අසත්‍ය වන්නේ කුමක්ද?"
        ))

    def test_negative_fact_inside_a_normal_question_does_not_route(self):
        self.assertFalse(is_si_negative_question(
            "වනාන්තර කැපීම නොකළ යුතු බවට නීති පැනවීමෙන් "
            "රජවරු බලාපොරොත්තු වූයේ කුමක්ද?"
        ))


if __name__ == "__main__":
    unittest.main()
