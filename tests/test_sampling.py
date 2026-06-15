from __future__ import annotations

import unittest

from bayesian_orchestrator.workflows.mmlu_bayesian_orchestrator import (
    MMLUQuestion,
    _split_question_ids,
    _stratified_sample_indices,
)


class SamplingTests(unittest.TestCase):
    def test_stratified_sample_is_exact_representative_and_reproducible(self) -> None:
        strata = ["large"] * 70 + ["medium"] * 20 + ["small"] * 10

        first = _stratified_sample_indices(strata, 40, seed=17)
        second = _stratified_sample_indices(strata, 40, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 40)
        self.assertEqual(len(set(first)), 40)
        selected = [strata[index] for index in first]
        self.assertEqual(selected.count("large"), 28)
        self.assertEqual(selected.count("medium"), 8)
        self.assertEqual(selected.count("small"), 4)

    def test_stratified_split_preserves_every_subject(self) -> None:
        questions = [
            MMLUQuestion(f"{subject}-{index}", subject, "Question?", ("A", "B"), "A")
            for subject in ("math", "law", "physics")
            for index in range(10)
        ]

        exploration, test = _split_question_ids(questions, seed=23, exploration_fraction=0.3, stratify=True)

        self.assertEqual(len(exploration), 9)
        self.assertEqual(len(test), 21)
        for subject in ("math", "law", "physics"):
            self.assertEqual(sum(question_id.startswith(subject) for question_id in exploration), 3)


if __name__ == "__main__":
    unittest.main()
