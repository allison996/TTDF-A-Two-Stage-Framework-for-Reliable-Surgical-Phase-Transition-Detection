import unittest

import numpy as np

from event_commit_metrics import (
    TransitionEvent,
    apply_legality_gate,
    dwell_commits,
    evaluate_commit_events,
)
from train_transition_reliability import prepost_probability_features


class CoreTest(unittest.TestCase):
    def test_dwell_legality_and_matching(self):
        labels = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 1])
        candidates = dwell_commits(labels, "v", dwell_duration=2)
        self.assertEqual(
            [(x.pair, x.time_index) for x in candidates],
            [((1, 0), 5), ((0, 1), 7)],
        )
        kept, rejected = apply_legality_gate(candidates, frozenset({(0, 1)}))
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(rejected), 1)
        result = evaluate_commit_events(
            [TransitionEvent("v", 0, 1, 5)], kept, tolerance=3
        )
        self.assertEqual(result["commit_f1"], 1.0)

    def test_posterior_cues_are_finite(self):
        probs = np.array([[0.9, 0.1]] * 4 + [[0.1, 0.9]] * 4)
        cues = prepost_probability_features(probs, 4, 7, 4)
        self.assertEqual(cues.shape, (2,))
        self.assertTrue(np.isfinite(cues).all())
        self.assertGreater(cues[0], 0)


if __name__ == "__main__":
    unittest.main()
