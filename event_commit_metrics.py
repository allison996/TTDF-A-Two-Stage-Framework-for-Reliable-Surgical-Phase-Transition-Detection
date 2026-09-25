"""Event-level metrics for online phase-transition commitments.

This file evaluates the decision output of a CAI state-commitment policy.
It is intentionally independent from model training:

Inputs
------
1. Ground-truth transition events:
   (video_id, phase_from, phase_to, time_index)
2. System commit events:
   (video_id, phase_from, phase_to, time_index)

Matching rule
-------------
A commit is a true positive when it has the same transition pair A->B as a
ground-truth transition in the same video and falls within +/- tolerance frames.
Each ground-truth transition can be matched at most once. Extra commits that hit
the same GT transition are counted as duplicates and as false positives.

Outputs
-------
commit_precision, commit_recall, commit_f1, false_commit_per_gt,
duplicate_per_gt, median_delay, and p90_delay.

At 1 fps, a tolerance of 10 frames corresponds to a +/-10 second relaxed window.

Optional policy gates
---------------------
The evaluator can also simulate a transition-legality gate. This suppresses
commits whose transition pair A->B is not allowed by a workflow graph. The graph
can be the Cholec80 workflow graph described by Funke et al. or estimated from
training annotations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


# Cholec80 workflow graph from Funke et al., "Metrics Matter in Surgical Phase
# Recognition", Fig. 2. The edge (A, B) means phase B can immediately follow A.
CHOLEC80_WORKFLOW_EDGES: frozenset[tuple[int, int]] = frozenset(
    {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (3, 5),
        (4, 5),
        (5, 4),
        (4, 6),
        (5, 6),
        (6, 5),
    }
)


@dataclass(frozen=True)
class TransitionEvent:
    """A point event representing a phase transition or system commit."""

    video_id: str
    phase_from: int
    phase_to: int
    time_index: int
    score: float | None = None

    @property
    def pair(self) -> tuple[int, int]:
        return self.phase_from, self.phase_to


def transition_indices(sequence: Sequence[int]) -> np.ndarray:
    """Return frame indices where a label sequence changes value."""

    labels = np.asarray(sequence)
    if labels.ndim != 1:
        raise ValueError("Expected a 1D label sequence.")
    if len(labels) < 2:
        return np.asarray([], dtype=np.int64)
    return np.flatnonzero(labels[1:] != labels[:-1]) + 1


def workflow_edges_from_feature_root(
    feature_root: Path,
    split: str = "train",
    temporal_stride: int = 1,
) -> frozenset[tuple[int, int]]:
    """Estimate legal transition edges from GT labels in a feature split.

    This is the data-derived alternative to using the fixed Cholec80 graph:
    every ground-truth transition A->B observed in the training split becomes a
    legal workflow edge. Self-transitions are not included because commit events
    are emitted only at label changes.
    """

    split_dir = feature_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Cannot find split directory: {split_dir}")

    edges: set[tuple[int, int]] = set()
    for path in sorted(split_dir.glob("video*.npz")):
        with np.load(path) as data:
            labels = data["labels"][::temporal_stride]
        for index in transition_indices(labels):
            edges.add((int(labels[index - 1]), int(labels[index])))

    if not edges:
        raise ValueError(f"No workflow edges found in {split_dir}")
    return frozenset(edges)


def events_from_sequence(sequence: Sequence[int], video_id: str) -> list[TransitionEvent]:
    """Convert every label change in a sequence into a transition event."""

    labels = np.asarray(sequence)
    return [
        TransitionEvent(
            video_id=video_id,
            phase_from=int(labels[index - 1]),
            phase_to=int(labels[index]),
            time_index=int(index),
        )
        for index in transition_indices(labels)
    ]


def raw_argmax_commits(predictions: Sequence[int], video_id: str) -> list[TransitionEvent]:
    """Baseline policy: commit immediately at every predicted phase change."""

    return events_from_sequence(predictions, video_id)


def raw_argmax_commits_with_scores(
    predictions: Sequence[int],
    probabilities: np.ndarray,
    video_id: str,
) -> list[TransitionEvent]:
    """Immediate commits scored by p(phase_to) at the transition frame."""

    labels = np.asarray(predictions)
    return [
        TransitionEvent(
            video_id=video_id,
            phase_from=int(labels[index - 1]),
            phase_to=int(labels[index]),
            time_index=int(index),
            score=float(probabilities[index, int(labels[index])]),
        )
        for index in transition_indices(labels)
    ]


def dwell_commits(
    predictions: Sequence[int],
    video_id: str,
    dwell_duration: int,
    probabilities: np.ndarray | None = None,
) -> list[TransitionEvent]:
    """Commit only if the new phase persists for ``dwell_duration`` frames.

    This simulates an online debounce/minimum-duration gate. For a candidate
    A->B at t_c, the policy waits k frames. If frames [t_c, t_c+k) all remain B,
    it emits a commit at t_c+k. Otherwise the candidate is suppressed.

    ``dwell_duration=0`` is equivalent to raw immediate commits.
    """

    if dwell_duration < 0:
        raise ValueError("dwell_duration must be non-negative.")
    if dwell_duration == 0:
        if probabilities is not None:
            return raw_argmax_commits_with_scores(predictions, probabilities, video_id)
        return raw_argmax_commits(predictions, video_id)

    labels = np.asarray(predictions)
    commits: list[TransitionEvent] = []
    for index in transition_indices(labels):
        phase_from = int(labels[index - 1])
        phase_to = int(labels[index])
        end = int(index) + dwell_duration
        if end > len(labels):
            continue
        if np.all(labels[index:end] == phase_to):
            score = None
            if probabilities is not None:
                score = float(probabilities[index:end, phase_to].mean())
            commits.append(
                TransitionEvent(
                    video_id=video_id,
                    phase_from=phase_from,
                    phase_to=phase_to,
                    time_index=end,
                    score=score,
                )
            )
    return commits


def apply_legality_gate(
    commits: Iterable[TransitionEvent],
    legal_edges: frozenset[tuple[int, int]] | None,
) -> tuple[list[TransitionEvent], list[TransitionEvent]]:
    """Suppress commits whose transition pair is not in the legal graph."""

    commit_list = list(commits)
    if legal_edges is None:
        return commit_list, []

    kept = [event for event in commit_list if event.pair in legal_edges]
    suppressed = [event for event in commit_list if event.pair not in legal_edges]
    return kept, suppressed


def apply_confidence_gate(
    commits: Iterable[TransitionEvent],
    confidence_threshold: float | None,
) -> tuple[list[TransitionEvent], list[TransitionEvent]]:
    """Suppress commits whose transition confidence is below a threshold."""

    commit_list = list(commits)
    if confidence_threshold is None:
        return commit_list, []
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1].")

    kept = [
        event
        for event in commit_list
        if event.score is not None and event.score >= confidence_threshold
    ]
    suppressed = [
        event
        for event in commit_list
        if event.score is None or event.score < confidence_threshold
    ]
    return kept, suppressed


def evaluate_commit_events(
    gt_events: Iterable[TransitionEvent],
    commit_events: Iterable[TransitionEvent],
    tolerance: int,
) -> dict:
    """Match system commits to GT transitions and compute event metrics."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    gt = sorted(gt_events, key=lambda event: (event.video_id, event.time_index))
    commits = sorted(commit_events, key=lambda event: (event.video_id, event.time_index))
    matched_gt: set[int] = set()
    true_positive = 0
    false_positive = 0
    duplicate = 0
    delays: list[int] = []
    matches = []
    false_commits = []

    for commit_index, commit in enumerate(commits):
        candidates = [
            (abs(commit.time_index - target.time_index), target_index, target)
            for target_index, target in enumerate(gt)
            if target.video_id == commit.video_id
            and target.pair == commit.pair
            and abs(commit.time_index - target.time_index) <= tolerance
        ]
        unmatched_candidates = [
            candidate for candidate in candidates if candidate[1] not in matched_gt
        ]

        if unmatched_candidates:
            _, target_index, target = min(
                unmatched_candidates, key=lambda item: (item[0], item[2].time_index)
            )
            matched_gt.add(target_index)
            true_positive += 1
            delay = int(commit.time_index - target.time_index)
            delays.append(delay)
            matches.append(
                {
                    "commit_index": commit_index,
                    "gt_index": target_index,
                    "video_id": commit.video_id,
                    "phase_from": commit.phase_from,
                    "phase_to": commit.phase_to,
                    "t_commit": commit.time_index,
                    "t_gt": target.time_index,
                    "delay": delay,
                    "duplicate": False,
                }
            )
        else:
            false_positive += 1
            is_duplicate = bool(candidates)
            duplicate += int(is_duplicate)
            false_commits.append(
                {
                    "commit_index": commit_index,
                    "video_id": commit.video_id,
                    "phase_from": commit.phase_from,
                    "phase_to": commit.phase_to,
                    "t_commit": commit.time_index,
                    "duplicate": is_duplicate,
                }
            )

    false_negative = len(gt) - len(matched_gt)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    commit_f1 = (
        2.0 * precision * recall / max(precision + recall, 1e-12)
        if true_positive > 0
        else 0.0
    )
    delay_array = np.asarray(delays, dtype=float)

    return {
        "num_gt": len(gt),
        "num_commits": len(commits),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "duplicates": duplicate,
        "commit_precision": precision,
        "commit_recall": recall,
        "commit_f1": commit_f1,
        "false_commit_per_gt": false_positive / max(len(gt), 1),
        "duplicate_per_gt": duplicate / max(len(gt), 1),
        "median_delay": float(np.median(delay_array)) if len(delay_array) else None,
        "p90_delay": float(np.percentile(delay_array, 90)) if len(delay_array) else None,
        "mean_delay": float(np.mean(delay_array)) if len(delay_array) else None,
        "matches": matches,
        "false_commits": false_commits,
    }


def evaluate_npz_directory(
    trajectory_dir: Path,
    policy: str,
    tolerance: int,
    dwell_duration: int = 0,
    legal_edges: frozenset[tuple[int, int]] | None = None,
    confidence_threshold: float | None = None,
) -> dict:
    """Evaluate all exported ``video*.npz`` trajectories in a split directory.

    Expected arrays per file are the ones exported by export_phase_predictions.py:
    ``labels`` and ``predictions``. The policy can be ``raw`` or ``dwell``.
    """

    gt_events: list[TransitionEvent] = []
    commit_events: list[TransitionEvent] = []
    suppressed_illegal: list[TransitionEvent] = []
    suppressed_low_confidence: list[TransitionEvent] = []
    for path in sorted(trajectory_dir.glob("*.npz")):
        with np.load(path) as data:
            labels = data["labels"]
            predictions = data["predictions"]
            probabilities = data["probabilities"].astype(np.float32)
        video_id = path.stem
        gt_events.extend(events_from_sequence(labels, video_id))
        if policy == "raw":
            video_commits = raw_argmax_commits_with_scores(
                predictions, probabilities, video_id
            )
        elif policy == "dwell":
            video_commits = dwell_commits(
                predictions, video_id, dwell_duration, probabilities=probabilities
            )
        else:
            raise ValueError(f"Unknown policy: {policy}")
        legal_commits, illegal_commits = apply_legality_gate(video_commits, legal_edges)
        confident_commits, low_confidence_commits = apply_confidence_gate(
            legal_commits, confidence_threshold
        )
        commit_events.extend(confident_commits)
        suppressed_illegal.extend(illegal_commits)
        suppressed_low_confidence.extend(low_confidence_commits)

    metrics = evaluate_commit_events(gt_events, commit_events, tolerance)
    metrics["suppressed_illegal_commits"] = len(suppressed_illegal)
    metrics["suppressed_low_confidence_commits"] = len(suppressed_low_confidence)
    metrics["legality_gate_enabled"] = legal_edges is not None
    metrics["confidence_threshold"] = confidence_threshold
    metrics["legal_edges"] = (
        [list(edge) for edge in sorted(legal_edges)] if legal_edges is not None else None
    )
    return metrics


def demo() -> dict:
    """Return a small hand-checkable example.

    GT transitions:
      video1: 0->1 at 5, 1->2 at 10
      video2: 0->1 at 4

    Commits:
      TP: 0->1 at 6 for video1, delay +1
      duplicate/FP: 0->1 at 7 for the same GT
      FP wrong timing: 1->2 at 20
      TP: 0->1 at 4 for video2, delay 0
      FP wrong target: 1->2 at 5 for video2

    With tolerance=2:
      TP=2, FP=3, FN=1, duplicates=1
      precision=2/5, recall=2/3, false/GT=3/3, median delay=0.5
    """

    gt_events = [
        TransitionEvent("video1", 0, 1, 5),
        TransitionEvent("video1", 1, 2, 10),
        TransitionEvent("video2", 0, 1, 4),
    ]
    commit_events = [
        TransitionEvent("video1", 0, 1, 6),
        TransitionEvent("video1", 0, 1, 7),
        TransitionEvent("video1", 1, 2, 20),
        TransitionEvent("video2", 0, 1, 4),
        TransitionEvent("video2", 1, 2, 5),
    ]
    return evaluate_commit_events(gt_events, commit_events, tolerance=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate event-level phase-transition commit metrics."
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Directory containing exported per-video .npz trajectories.",
    )
    parser.add_argument("--policy", choices=["raw", "dwell"], default="raw")
    parser.add_argument("--tolerance", type=int, default=10)
    parser.add_argument("--dwell-duration", type=int, default=10)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help=(
            "Optional MSP confidence gate. For dwell policies, confidence is mean "
            "p(phase_to) over the dwell window."
        ),
    )
    parser.add_argument(
        "--legality-gate",
        choices=["none", "cholec80", "train"],
        default="none",
        help=(
            "Optional transition-legality gate. 'cholec80' uses the Funke et al. "
            "workflow graph; 'train' estimates legal edges from feature_root/train."
        ),
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=None,
        help="Feature root used when --legality-gate=train.",
    )
    parser.add_argument(
        "--legal-split",
        default="train",
        help="Feature split used to estimate legal edges when --legality-gate=train.",
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=1,
        help="Temporal stride for reading GT labels when --legality-gate=train.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in hand-checkable example instead of loading files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        metrics = demo()
    else:
        if args.trajectory_dir is None:
            raise SystemExit("--trajectory-dir is required unless --demo is set.")
        if args.legality_gate == "none":
            legal_edges = None
        elif args.legality_gate == "cholec80":
            legal_edges = CHOLEC80_WORKFLOW_EDGES
        else:
            if args.feature_root is None:
                raise SystemExit("--feature-root is required with --legality-gate=train.")
            legal_edges = workflow_edges_from_feature_root(
                args.feature_root,
                split=args.legal_split,
                temporal_stride=args.temporal_stride,
            )
        metrics = evaluate_npz_directory(
            args.trajectory_dir,
            policy=args.policy,
            tolerance=args.tolerance,
            dwell_duration=args.dwell_duration,
            legal_edges=legal_edges,
            confidence_threshold=args.confidence_threshold,
        )

    serializable = {
        key: value
        for key, value in metrics.items()
        if key not in {"matches", "false_commits"}
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as file:
            json.dump(metrics, file, indent=2)
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
