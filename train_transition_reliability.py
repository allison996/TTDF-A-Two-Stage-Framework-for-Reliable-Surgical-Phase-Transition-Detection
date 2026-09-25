"""Train the S2 candidate-level transition reliability head.

This script trains a lightweight verifier on exported recognizer trajectories.
It does not retrain the frame-wise phase recognizer.  The training unit is a
candidate transition A->B that survived the S1 gates, and the target is whether
that candidate commit matches a ground-truth transition under the event-level
matching protocol.

Typical workflow:

1. Export full-video recognizer outputs for train/val/test:

   python export_phase_predictions.py best.pt --splits train val test \
       --output-dir outputs/dwell_5 --persistent-duration 5

2. Train the S2 verifier:

   python train_transition_reliability.py \
       --train-dir outputs/dwell_5/train \
       --val-dir outputs/dwell_5/val \
       --test-dir outputs/dwell_5/test \
       --feature-root /data/Cholec80/dinov2_cls_features \
       --dwell-duration 5 --tolerance 30 --feature-set pa_pb
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from event_commit_metrics import (
    CHOLEC80_WORKFLOW_EDGES,
    TransitionEvent,
    evaluate_commit_events,
    events_from_sequence,
    transition_indices,
    workflow_edges_from_feature_root,
)
from models.transition_reliability import (
    TransitionReliabilityHead,
    correctness_ranking_loss,
    sigmoid_focal_bce_loss,
    smooth_l1_offset_loss,
)


@dataclass(frozen=True)
class CandidateExample:
    """One S1-surviving candidate transition used by the S2 verifier."""

    window: np.ndarray
    mask: np.ndarray
    label: float
    event: TransitionEvent
    video_id: str
    t_candidate: int
    t_commit: int
    msp_score: float
    # Frames from t_candidate to the matched ground-truth transition instant.
    # Only meaningful when label == 1; 0.0 otherwise and excluded from the
    # offset loss via the label-derived positive mask.
    offset_target: float = 0.0


def transition_identity_dim(transition_identity: str, num_phases: int) -> int:
    if transition_identity == "none":
        return 0
    if transition_identity == "phases":
        return 2 * num_phases
    if transition_identity == "edge":
        return num_phases * num_phases
    raise ValueError(f"Unknown transition_identity: {transition_identity}")


def log_evidence_dim(log_evidence: str) -> int:
    if log_evidence == "none":
        return 0
    if log_evidence in {"instant", "accum"}:
        return 1
    if log_evidence == "both":
        return 2
    raise ValueError(f"Unknown log_evidence: {log_evidence}")


def prototype_evidence_dim(prototype_evidence: str) -> int:
    if prototype_evidence == "none":
        return 0
    if prototype_evidence == "phase_margin":
        return 1
    raise ValueError(f"Unknown prototype_evidence: {prototype_evidence}")


def reliability_cue_dim(
    competing_phase_cues: bool = False,
    prepost_prob_cues: bool = False,
    prepost_prob_cue_mode: str = "both",
    visual_change_cue: bool = False,
) -> int:
    dim = 0
    if competing_phase_cues:
        dim += 3
    if prepost_prob_cues:
        if prepost_prob_cue_mode == "both":
            dim += 2
        elif prepost_prob_cue_mode in {"l1", "js"}:
            dim += 1
        else:
            raise ValueError(f"Unknown prepost_prob_cue_mode: {prepost_prob_cue_mode}")
    if visual_change_cue:
        dim += 1
    return dim


def feature_dim(
    feature_set: str,
    transition_identity: str = "none",
    num_phases: int = 7,
    log_evidence: str = "none",
    prototype_evidence: str = "none",
    competing_phase_cues: bool = False,
    prepost_prob_cues: bool = False,
    prepost_prob_cue_mode: str = "both",
    visual_change_cue: bool = False,
) -> int:
    if feature_set == "none":
        base_dim = 0
    elif feature_set == "pa_pb":
        base_dim = 2
    elif feature_set == "prob5":
        base_dim = 5
    elif feature_set == "prob7":
        base_dim = 7
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    return (
        base_dim
        + transition_identity_dim(transition_identity, num_phases)
        + log_evidence_dim(log_evidence)
        + prototype_evidence_dim(prototype_evidence)
        + reliability_cue_dim(
            competing_phase_cues,
            prepost_prob_cues,
            prepost_prob_cue_mode,
            visual_change_cue,
        )
    )


def transition_identity_features(
    phase_from: int,
    phase_to: int,
    transition_identity: str,
    num_phases: int,
) -> np.ndarray:
    if transition_identity == "none":
        return np.zeros((0,), dtype=np.float32)
    if transition_identity == "phases":
        features = np.zeros((2 * num_phases,), dtype=np.float32)
        if 0 <= phase_from < num_phases:
            features[phase_from] = 1.0
        if 0 <= phase_to < num_phases:
            features[num_phases + phase_to] = 1.0
        return features
    if transition_identity == "edge":
        features = np.zeros((num_phases * num_phases,), dtype=np.float32)
        if 0 <= phase_from < num_phases and 0 <= phase_to < num_phases:
            features[phase_from * num_phases + phase_to] = 1.0
        return features
    raise ValueError(f"Unknown transition_identity: {transition_identity}")


def log_evidence_features(
    log_delta: float,
    accumulated_log_evidence: float,
    log_evidence: str,
) -> np.ndarray:
    if log_evidence == "none":
        return np.zeros((0,), dtype=np.float32)
    if log_evidence == "instant":
        return np.asarray([log_delta], dtype=np.float32)
    if log_evidence == "accum":
        return np.asarray([accumulated_log_evidence], dtype=np.float32)
    if log_evidence == "both":
        return np.asarray([log_delta, accumulated_log_evidence], dtype=np.float32)
    raise ValueError(f"Unknown log_evidence: {log_evidence}")


def competing_phase_features(
    probabilities: np.ndarray,
    phase_from: int,
    phase_to: int,
) -> np.ndarray:
    """Summarize whether B beats all competing phases, not only A."""

    probs = probabilities.astype(np.float32)
    num_classes = len(probs)
    if num_classes <= 1:
        return np.zeros((3,), dtype=np.float32)

    other_ab_mask = np.ones((num_classes,), dtype=bool)
    if 0 <= phase_from < num_classes:
        other_ab_mask[phase_from] = False
    if 0 <= phase_to < num_classes:
        other_ab_mask[phase_to] = False
    p_other_ab = float(probs[other_ab_mask].max()) if other_ab_mask.any() else 0.0

    non_b_mask = np.ones((num_classes,), dtype=bool)
    if 0 <= phase_to < num_classes:
        non_b_mask[phase_to] = False
    p_non_b = float(probs[non_b_mask].max()) if non_b_mask.any() else 0.0
    p_b = float(probs[phase_to]) if 0 <= phase_to < num_classes else 0.0

    return np.asarray(
        [
            p_other_ab,
            p_b - p_non_b,
            p_b - p_other_ab,
        ],
        dtype=np.float32,
    )


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.clip(left.astype(np.float64), 1e-8, 1.0)
    right = np.clip(right.astype(np.float64), 1e-8, 1.0)
    left = left / left.sum()
    right = right / right.sum()
    midpoint = 0.5 * (left + right)
    kl_left = np.sum(left * (np.log(left) - np.log(midpoint)))
    kl_right = np.sum(right * (np.log(right) - np.log(midpoint)))
    return float(0.5 * (kl_left + kl_right))


def prepost_ranges(
    num_frames: int,
    t_candidate: int,
    t_commit: int,
    left_context: int,
) -> tuple[slice, slice]:
    pre_start = max(0, int(t_candidate) - max(left_context, 1))
    pre_stop = max(pre_start + 1, min(num_frames, int(t_candidate)))
    post_start = min(max(0, int(t_candidate)), num_frames - 1)
    post_stop = max(post_start + 1, min(num_frames, int(t_commit) + 1))
    return slice(pre_start, pre_stop), slice(post_start, post_stop)


def prepost_probability_features(
    probabilities: np.ndarray,
    t_candidate: int,
    t_commit: int,
    left_context: int,
    mode: str = "both",
) -> np.ndarray:
    """Candidate-level probability change cue, repeated across the window."""

    num_frames = len(probabilities)
    if num_frames == 0:
        return np.zeros((2,), dtype=np.float32)
    pre_slice, post_slice = prepost_ranges(
        num_frames, t_candidate, t_commit, left_context
    )
    pre_mean = probabilities[pre_slice].mean(axis=0)
    post_mean = probabilities[post_slice].mean(axis=0)
    l1_change = float(np.abs(post_mean - pre_mean).sum())
    js_change = js_divergence(pre_mean, post_mean)
    if mode == "l1":
        return np.asarray([l1_change], dtype=np.float32)
    if mode == "js":
        return np.asarray([js_change], dtype=np.float32)
    if mode != "both":
        raise ValueError(f"Unknown prepost probability cue mode: {mode}")
    return np.asarray([l1_change, js_change], dtype=np.float32)


def visual_change_feature(
    visual_features: np.ndarray | None,
    t_candidate: int,
    t_commit: int,
    left_context: int,
) -> np.ndarray:
    """Candidate-level pre/post visual change magnitude from frozen features."""

    if visual_features is None or len(visual_features) == 0:
        return np.zeros((1,), dtype=np.float32)
    num_frames = len(visual_features)
    pre_slice, post_slice = prepost_ranges(
        num_frames, t_candidate, t_commit, left_context
    )
    pre_mean = visual_features[pre_slice].mean(axis=0)
    post_mean = visual_features[post_slice].mean(axis=0)
    denom = float(np.linalg.norm(pre_mean) * np.linalg.norm(post_mean))
    if denom <= 1e-12:
        return np.zeros((1,), dtype=np.float32)
    cosine_distance = 1.0 - float(np.dot(pre_mean, post_mean) / denom)
    return np.asarray([cosine_distance], dtype=np.float32)


def normalize_rows(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32)
    return features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)


def resolve_feature_dir(feature_root: Path, split: str) -> Path:
    split_dir = feature_root / split
    if list(split_dir.glob("video*.npz")):
        return split_dir
    if list(feature_root.glob("video*.npz")):
        return feature_root
    raise FileNotFoundError(f"No video*.npz found in {split_dir} or {feature_root}")


def build_phase_prototypes(feature_dir: Path) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Build train-split visual prototypes from normalized DINO features."""

    sums: dict[int, np.ndarray] = {}
    counts: defaultdict[int, int] = defaultdict(int)
    for path in sorted(feature_dir.glob("video*.npz")):
        with np.load(path) as data:
            features = normalize_rows(data["features"])
            labels = data["labels"].astype(np.int64)
        for cls in np.unique(labels):
            cls = int(cls)
            mask = labels == cls
            total = features[mask].sum(axis=0).astype(np.float64)
            if cls not in sums:
                sums[cls] = total
            else:
                sums[cls] += total
            counts[cls] += int(mask.sum())

    prototypes = {}
    for cls, total in sums.items():
        prototype = total / max(counts[cls], 1)
        prototype = prototype / np.clip(np.linalg.norm(prototype), 1e-12, None)
        prototypes[cls] = prototype.astype(np.float32)
    return prototypes, dict(counts)


def prototype_evidence_features(
    visual_features: np.ndarray | None,
    index: int,
    phase_from: int,
    phase_to: int,
    prototypes: dict[int, np.ndarray] | None,
    prototype_evidence: str,
) -> np.ndarray:
    if prototype_evidence == "none":
        return np.zeros((0,), dtype=np.float32)
    if prototype_evidence != "phase_margin":
        raise ValueError(f"Unknown prototype_evidence: {prototype_evidence}")
    if visual_features is None or prototypes is None:
        return np.asarray([0.0], dtype=np.float32)
    proto_from = prototypes.get(int(phase_from))
    proto_to = prototypes.get(int(phase_to))
    if proto_from is None or proto_to is None:
        return np.asarray([0.0], dtype=np.float32)
    feature = visual_features[int(np.clip(index, 0, len(visual_features) - 1))]
    margin = float(np.dot(feature, proto_to) - np.dot(feature, proto_from))
    return np.asarray([margin], dtype=np.float32)


def candidate_window_features(
    probabilities: np.ndarray,
    visual_features: np.ndarray | None,
    phase_from: int,
    phase_to: int,
    t_candidate: int,
    t_commit: int,
    left_context: int,
    feature_set: str,
    transition_identity: str,
    num_phases: int,
    log_evidence: str,
    prototype_evidence: str,
    phase_prototypes: dict[int, np.ndarray] | None,
    competing_phase_cues: bool,
    prepost_prob_cues: bool,
    prepost_prob_cue_mode: str,
    visual_change_cue: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a short online evidence window for one candidate.

    The window ends at the dwell commit time, so it uses only frames available
    when the online policy could make the S2 decision.
    """

    probabilities = probabilities.astype(np.float32)
    num_frames, num_classes = probabilities.shape
    start = int(t_candidate) - left_context + 1
    # Include the frame at t_commit: the policy emits its decision at this time,
    # so the current recognizer output is available to the S2 verifier.
    stop = int(t_commit) + 1
    length = stop - start
    if length <= 0:
        raise ValueError("candidate window length must be positive.")

    identity = transition_identity_features(
        phase_from, phase_to, transition_identity, num_phases
    )
    window = np.zeros(
        (
            length,
            feature_dim(
                feature_set,
                transition_identity,
                num_phases,
                log_evidence,
                prototype_evidence,
                competing_phase_cues,
                prepost_prob_cues,
                prepost_prob_cue_mode,
                visual_change_cue,
            ),
        ),
        dtype=np.float32,
    )
    mask = np.zeros((length,), dtype=bool)
    previous_p_b = 0.0
    previous_entropy = 0.0
    accumulated_log_evidence = 0.0
    global_cues = []
    if prepost_prob_cues:
        global_cues.append(
            prepost_probability_features(
                probabilities,
                t_candidate,
                t_commit,
                left_context,
                mode=prepost_prob_cue_mode,
            )
        )
    if visual_change_cue:
        global_cues.append(
            visual_change_feature(
                visual_features, t_candidate, t_commit, left_context
            )
        )
    global_cues_array = (
        np.concatenate(global_cues).astype(np.float32)
        if global_cues
        else np.zeros((0,), dtype=np.float32)
    )

    for row, index in enumerate(range(start, stop)):
        if index < 0 or index >= num_frames:
            continue
        probs = probabilities[index]
        mask[row] = True
        p_a = float(probs[phase_from])
        p_b = float(probs[phase_to])
        log_delta = float(
            np.log(np.clip(p_b, 1e-8, 1.0)) - np.log(np.clip(p_a, 1e-8, 1.0))
        )
        accumulated_log_evidence = max(0.0, accumulated_log_evidence + log_delta)
        log_features = log_evidence_features(
            log_delta, accumulated_log_evidence, log_evidence
        )
        prototype_features = prototype_evidence_features(
            visual_features,
            index,
            phase_from,
            phase_to,
            phase_prototypes,
            prototype_evidence,
        )
        competing_features = (
            competing_phase_features(probs, phase_from, phase_to)
            if competing_phase_cues
            else np.zeros((0,), dtype=np.float32)
        )
        reliability_cues = np.concatenate(
            (competing_features, global_cues_array)
        ).astype(np.float32)
        if feature_set == "none":
            window[row] = np.concatenate(
                (
                    identity,
                    log_features,
                    prototype_features,
                    reliability_cues,
                )
            )
            continue

        if feature_set == "pa_pb":
            window[row] = np.concatenate(
                (
                    [p_a, p_b],
                    identity,
                    log_features,
                    prototype_features,
                    reliability_cues,
                )
            )
            continue

        max_prob = float(probs.max())
        entropy = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())
        if num_classes >= 2:
            top2 = np.partition(probs, -2)[-2:]
            margin = float(top2[-1] - top2[-2])
        else:
            margin = max_prob
        if feature_set == "prob5":
            window[row] = np.concatenate(
                (
                    [p_a, p_b, max_prob, entropy, margin],
                    identity,
                    log_features,
                    prototype_features,
                    reliability_cues,
                )
            )
            continue

        delta_p_b = p_b - previous_p_b
        delta_entropy = entropy - previous_entropy
        window[row] = np.concatenate(
            (
                [
                    p_a,
                    p_b,
                    max_prob,
                    entropy,
                    margin,
                    delta_p_b,
                    delta_entropy,
                ],
                identity,
                log_features,
                prototype_features,
                reliability_cues,
            )
        )
        previous_p_b = p_b
        previous_entropy = entropy

    return window, mask


def s1_surviving_candidates(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    visual_features: np.ndarray | None,
    video_id: str,
    dwell_duration: int,
    legal_edges: frozenset[tuple[int, int]] | None,
    left_context: int,
    feature_set: str,
    transition_identity: str,
    num_phases: int,
    log_evidence: str,
    prototype_evidence: str,
    phase_prototypes: dict[int, np.ndarray] | None,
    competing_phase_cues: bool,
    prepost_prob_cues: bool,
    prepost_prob_cue_mode: str,
    visual_change_cue: bool,
) -> list[dict]:
    """Generate dwell+legality surviving candidates from one trajectory."""

    candidates: list[dict] = []
    for index in transition_indices(predictions):
        phase_from = int(predictions[index - 1])
        phase_to = int(predictions[index])
        t_candidate = int(index)
        t_commit = t_candidate + dwell_duration
        if t_commit > len(predictions):
            continue
        if dwell_duration > 0 and not np.all(
            predictions[t_candidate:t_commit] == phase_to
        ):
            continue
        if legal_edges is not None and (phase_from, phase_to) not in legal_edges:
            continue
        window, mask = candidate_window_features(
            probabilities,
            visual_features,
            phase_from,
            phase_to,
            t_candidate,
            t_commit,
            left_context,
            feature_set,
            transition_identity,
            num_phases,
            log_evidence,
            prototype_evidence,
            phase_prototypes,
            competing_phase_cues,
            prepost_prob_cues,
            prepost_prob_cue_mode,
            visual_change_cue,
        )
        dwell_slice = probabilities[t_candidate:t_commit, phase_to]
        msp_score = float(dwell_slice.mean()) if len(dwell_slice) else float(
            probabilities[t_candidate, phase_to]
        )
        candidates.append(
            {
                "window": window,
                "mask": mask,
                "event": TransitionEvent(
                    video_id=video_id,
                    phase_from=phase_from,
                    phase_to=phase_to,
                    time_index=t_commit,
                    score=msp_score,
                ),
                "video_id": video_id,
                "t_candidate": t_candidate,
                "t_commit": t_commit,
                "msp_score": msp_score,
            }
        )
    return candidates


def label_candidates(
    gt_events: list[TransitionEvent],
    candidates: list[dict],
    tolerance: int,
) -> list[CandidateExample]:
    """Assign commit-correct labels using the same one-to-one event matching."""

    gt = sorted(gt_events, key=lambda event: (event.video_id, event.time_index))
    candidate_order = sorted(
        range(len(candidates)),
        key=lambda i: (candidates[i]["event"].video_id, candidates[i]["event"].time_index),
    )
    matched_gt: set[int] = set()
    labels = np.zeros((len(candidates),), dtype=np.float32)
    offset_targets = np.zeros((len(candidates),), dtype=np.float32)

    for candidate_index in candidate_order:
        commit = candidates[candidate_index]["event"]
        possible = [
            (abs(commit.time_index - target.time_index), target_index, target)
            for target_index, target in enumerate(gt)
            if target.video_id == commit.video_id
            and target.pair == commit.pair
            and abs(commit.time_index - target.time_index) <= tolerance
            and target_index not in matched_gt
        ]
        if not possible:
            continue
        _, target_index, matched_target = min(
            possible, key=lambda item: (item[0], item[2].time_index)
        )
        matched_gt.add(target_index)
        labels[candidate_index] = 1.0
        # Offset target is relative to t_candidate (the trigger frame), the
        # same reference point used by TransitionReliabilityHead's
        # window_offset_start, so the two are directly comparable.
        offset_targets[candidate_index] = float(
            matched_target.time_index - candidates[candidate_index]["t_candidate"]
        )

    examples = []
    for candidate, label, offset_target in zip(candidates, labels, offset_targets):
        examples.append(
            CandidateExample(
                window=candidate["window"],
                mask=candidate["mask"],
                label=float(label),
                event=candidate["event"],
                video_id=candidate["video_id"],
                t_candidate=candidate["t_candidate"],
                t_commit=candidate["t_commit"],
                msp_score=candidate["msp_score"],
                offset_target=float(offset_target),
            )
        )
    return examples


class TransitionCandidateDataset(Dataset):
    """Candidate dataset built from exported full-video recognizer trajectories."""

    def __init__(
        self,
        trajectory_dir: Path,
        dwell_duration: int,
        tolerance: int,
        legal_edges: frozenset[tuple[int, int]] | None,
        left_context: int,
        feature_set: str,
        transition_identity: str,
        num_phases: int,
        log_evidence: str,
        prototype_evidence: str,
        competing_phase_cues: bool = False,
        prepost_prob_cues: bool = False,
        prepost_prob_cue_mode: str = "both",
        visual_change_cue: bool = False,
        feature_dir: Path | None = None,
        phase_prototypes: dict[int, np.ndarray] | None = None,
    ):
        self.trajectory_dir = Path(trajectory_dir)
        if not self.trajectory_dir.exists():
            raise FileNotFoundError(f"Cannot find trajectory dir: {self.trajectory_dir}")
        if prototype_evidence != "none" and (feature_dir is None or phase_prototypes is None):
            raise ValueError(
                "feature_dir and phase_prototypes are required when prototype evidence is enabled."
            )
        if visual_change_cue and feature_dir is None:
            raise ValueError("feature_dir is required when --visual-change-cue is enabled.")
        self.feature_dir = Path(feature_dir) if feature_dir is not None else None
        self.gt_events: list[TransitionEvent] = []
        raw_candidates: list[dict] = []

        for path in sorted(self.trajectory_dir.glob("*.npz")):
            with np.load(path) as data:
                labels = data["labels"].astype(np.int64)
                predictions = data["predictions"].astype(np.int64)
                probabilities = data["probabilities"].astype(np.float32)
            video_id = path.stem
            visual_features = None
            if prototype_evidence != "none" or visual_change_cue:
                feature_path = self.feature_dir / f"{video_id}.npz"
                if not feature_path.exists():
                    raise FileNotFoundError(f"Cannot find feature file: {feature_path}")
                with np.load(feature_path) as feature_data:
                    visual_features = normalize_rows(feature_data["features"])
            self.gt_events.extend(events_from_sequence(labels, video_id))
            raw_candidates.extend(
                s1_surviving_candidates(
                    predictions,
                    probabilities,
                    visual_features,
                    video_id,
                    dwell_duration=dwell_duration,
                    legal_edges=legal_edges,
                    left_context=left_context,
                    feature_set=feature_set,
                    transition_identity=transition_identity,
                    num_phases=num_phases,
                    log_evidence=log_evidence,
                    prototype_evidence=prototype_evidence,
                    phase_prototypes=phase_prototypes,
                    competing_phase_cues=competing_phase_cues,
                    prepost_prob_cues=prepost_prob_cues,
                    prepost_prob_cue_mode=prepost_prob_cue_mode,
                    visual_change_cue=visual_change_cue,
                )
            )

        if not raw_candidates:
            raise ValueError(f"No S1-surviving candidates found in {trajectory_dir}")
        self.examples = label_candidates(self.gt_events, raw_candidates, tolerance)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "window": torch.from_numpy(example.window),
            "mask": torch.from_numpy(example.mask),
            "label": torch.tensor(example.label, dtype=torch.float32),
            "offset_target": torch.tensor(example.offset_target, dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }

    @property
    def positive_count(self) -> int:
        return int(sum(example.label > 0.5 for example in self.examples))

    @property
    def negative_count(self) -> int:
        return len(self.examples) - self.positive_count

    @property
    def events(self) -> list[TransitionEvent]:
        return [example.event for example in self.examples]

    @property
    def msp_scores(self) -> np.ndarray:
        return np.asarray([example.msp_score for example in self.examples], dtype=np.float32)


def collate_candidates(batch: list[dict]) -> dict:
    return {
        "window": torch.stack([item["window"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]).bool(),
        "label": torch.stack([item["label"] for item in batch]),
        "offset_target": torch.stack([item["offset_target"] for item in batch]),
        "index": torch.stack([item["index"] for item in batch]),
    }


@torch.inference_mode()
def predict_scores(
    model: TransitionReliabilityHead,
    dataset: TransitionCandidateDataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_candidates,
    )
    scores = np.zeros((len(dataset),), dtype=np.float32)
    for batch in loader:
        window = batch["window"].to(device)
        mask = batch["mask"].to(device)
        output = model(window, mask)
        scores[batch["index"].numpy()] = output["probability"].cpu().numpy()
    return scores


def evaluate_scores(
    dataset: TransitionCandidateDataset,
    scores: np.ndarray,
    threshold: float,
    tolerance: int,
) -> dict:
    commits = [
        TransitionEvent(
            video_id=example.event.video_id,
            phase_from=example.event.phase_from,
            phase_to=example.event.phase_to,
            time_index=example.event.time_index,
            score=float(score),
        )
        for example, score in zip(dataset.examples, scores)
        if float(score) >= threshold
    ]
    metrics = evaluate_commit_events(dataset.gt_events, commits, tolerance)
    metrics["threshold"] = float(threshold)
    return metrics


def candidate_eval_evidence(example: CandidateExample) -> dict:
    """Extract interpretable evidence at the S2 evaluation time."""

    valid_rows = np.flatnonzero(example.mask)
    if len(valid_rows) == 0:
        row = example.window[-1]
    else:
        row = example.window[int(valid_rows[-1])]
    p_a = float(row[0]) if len(row) > 0 else 0.0
    p_b = float(row[1]) if len(row) > 1 else 0.0
    instant_log_evidence = float(
        np.log(np.clip(p_b, 1e-8, 1.0)) - np.log(np.clip(p_a, 1e-8, 1.0))
    )

    accum = 0.0
    for valid_index in valid_rows:
        valid_row = example.window[int(valid_index)]
        pa_i = float(valid_row[0]) if len(valid_row) > 0 else 0.0
        pb_i = float(valid_row[1]) if len(valid_row) > 1 else 0.0
        delta = float(
            np.log(np.clip(pb_i, 1e-8, 1.0)) - np.log(np.clip(pa_i, 1e-8, 1.0))
        )
        accum = max(0.0, accum + delta)

    return {
        "p_a": p_a,
        "p_b": p_b,
        "instant_log_evidence": instant_log_evidence,
        "accum_log_evidence": float(accum),
    }


def export_trigger_table(
    dataset: TransitionCandidateDataset,
    scores: np.ndarray,
    threshold: float,
    tolerance: int,
    output_path: Path,
    seed: int,
    dwell_duration: int,
    legal_edges: frozenset[tuple[int, int]] | None,
) -> None:
    """Save one row per S1 candidate with S2 decision and event-match status."""

    committed = []
    committed_example_indices = []
    for example_index, (example, score) in enumerate(zip(dataset.examples, scores)):
        if float(score) < threshold:
            continue
        committed_example_indices.append(example_index)
        committed.append(
            TransitionEvent(
                video_id=example.event.video_id,
                phase_from=example.event.phase_from,
                phase_to=example.event.phase_to,
                time_index=example.event.time_index,
                score=float(score),
            )
        )

    match_metrics = evaluate_commit_events(dataset.gt_events, committed, tolerance)
    commit_status = {}
    gt_by_key = {
        (event.video_id, event.phase_from, event.phase_to, event.time_index): gt_index
        for gt_index, event in enumerate(
            sorted(dataset.gt_events, key=lambda item: (item.video_id, item.time_index))
        )
    }
    for match in match_metrics.get("matches", []):
        commit_index = int(match["commit_index"])
        example_index = committed_example_indices[commit_index]
        status = {
            "match_status": "TP",
            "gt_match_id": int(match["gt_index"]),
            "gt_time": int(match["t_gt"]),
            "delay": int(match["delay"]),
        }
        commit_status[example_index] = status

    for false_commit in match_metrics.get("false_commits", []):
        commit_index = int(false_commit["commit_index"])
        example_index = committed_example_indices[commit_index]
        is_duplicate = bool(false_commit["duplicate"])
        commit_status[example_index] = {
            "match_status": "duplicate" if is_duplicate else "FP",
            "gt_match_id": "",
            "gt_time": "",
            "delay": "",
        }

    fieldnames = [
        "seed",
        "video_id",
        "candidate_id",
        "phase_from",
        "phase_to",
        "candidate_onset",
        "evaluation_time",
        "commit_time",
        "dwell",
        "legal",
        "s1_confidence",
        "reliability_score",
        "threshold",
        "decision",
        "gt_match_id",
        "gt_time",
        "match_status",
        "delay",
        "p_a",
        "p_b",
        "instant_log_evidence",
        "accum_log_evidence",
        "reason",
    ]

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for candidate_id, (example, score) in enumerate(zip(dataset.examples, scores)):
            decision = "commit" if float(score) >= threshold else "suppress"
            legal = legal_edges is None or example.event.pair in legal_edges
            evidence = candidate_eval_evidence(example)

            if decision == "commit":
                status = commit_status.get(
                    candidate_id,
                    {
                        "match_status": "FP",
                        "gt_match_id": "",
                        "gt_time": "",
                        "delay": "",
                    },
                )
                reason = f"dwell_pass; legal={int(legal)}; score={float(score):.4f}>={threshold:.4f}"
            else:
                status = {
                    "match_status": (
                        "suppressed-positive"
                        if example.label > 0.5
                        else "suppressed-negative"
                    ),
                    "gt_match_id": "",
                    "gt_time": "",
                    "delay": "",
                }
                if example.label > 0.5:
                    gt_time = int(example.t_candidate + example.offset_target)
                    key = (
                        example.event.video_id,
                        example.event.phase_from,
                        example.event.phase_to,
                        gt_time,
                    )
                    status["gt_match_id"] = gt_by_key.get(key, "")
                    status["gt_time"] = gt_time
                    status["delay"] = int(example.t_commit - gt_time)
                reason = f"suppressed; score={float(score):.4f}<{threshold:.4f}"

            writer.writerow(
                {
                    "seed": int(seed),
                    "video_id": example.event.video_id,
                    "candidate_id": int(candidate_id),
                    "phase_from": int(example.event.phase_from),
                    "phase_to": int(example.event.phase_to),
                    "candidate_onset": int(example.t_candidate),
                    "evaluation_time": int(example.t_commit),
                    "commit_time": int(example.t_commit) if decision == "commit" else "",
                    "dwell": int(dwell_duration),
                    "legal": int(legal),
                    "s1_confidence": float(example.msp_score),
                    "reliability_score": float(score),
                    "threshold": float(threshold),
                    "decision": decision,
                    "gt_match_id": status["gt_match_id"],
                    "gt_time": status["gt_time"],
                    "match_status": status["match_status"],
                    "delay": status["delay"],
                    "p_a": evidence["p_a"],
                    "p_b": evidence["p_b"],
                    "instant_log_evidence": evidence["instant_log_evidence"],
                    "accum_log_evidence": evidence["accum_log_evidence"],
                    "reason": reason,
                }
            )


def sweep_thresholds(
    dataset: TransitionCandidateDataset,
    scores: np.ndarray,
    tolerance: int,
    recall_retention: float,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict, list[dict]]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    base_metrics = evaluate_scores(dataset, np.ones_like(scores), 0.0, tolerance)
    min_recall = recall_retention * base_metrics["commit_recall"]
    rows = []
    feasible = []
    for threshold in thresholds:
        metrics = evaluate_scores(dataset, scores, float(threshold), tolerance)
        metrics["base_recall"] = base_metrics["commit_recall"]
        metrics["min_recall"] = min_recall
        rows.append(metrics)
        if metrics["commit_recall"] >= min_recall:
            feasible.append(metrics)

    if feasible:
        best = min(
            feasible,
            key=lambda item: (
                item["false_commit_per_gt"],
                -item["commit_precision"],
                -item["commit_f1"],
            ),
        )
    else:
        best = max(rows, key=lambda item: item["commit_f1"])
    return float(best["threshold"]), best, rows


def train_one_epoch(
    model: TransitionReliabilityHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    focal_alpha: float,
    focal_gamma: float,
    ranking_weight: float,
    ranking_margin: float,
    offset_weight: float,
    offset_beta: float,
) -> dict:
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_rank = 0.0
    total_offset = 0.0
    total_count = 0
    for batch in loader:
        window = batch["window"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        offset_target = batch["offset_target"].to(device)
        output = model(window, mask)
        focal = sigmoid_focal_bce_loss(
            output["logit"], label, alpha=focal_alpha, gamma=focal_gamma
        )
        rank = correctness_ranking_loss(
            output["probability"], label, margin=ranking_margin
        )
        loss = focal + ranking_weight * rank

        offset_loss = focal.new_tensor(0.0)
        if offset_weight > 0 and "offset" in output:
            offset_loss = smooth_l1_offset_loss(
                output["offset"], offset_target, positive_mask=label, beta=offset_beta
            )
            loss = loss + offset_weight * offset_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = int(label.numel())
        total_count += batch_size
        total_loss += float(loss.item()) * batch_size
        total_bce += float(focal.item()) * batch_size
        total_rank += float(rank.item()) * batch_size
        total_offset += float(offset_loss.item()) * batch_size

    return {
        "loss": total_loss / max(total_count, 1),
        "focal_loss": total_bce / max(total_count, 1),
        "ranking_loss": total_rank / max(total_count, 1),
        "offset_loss": total_offset / max(total_count, 1),
    }


def compact_metrics(metrics: dict) -> dict:
    keys = [
        "num_gt",
        "num_commits",
        "tp",
        "fp",
        "fn",
        "duplicates",
        "commit_precision",
        "commit_recall",
        "commit_f1",
        "false_commit_per_gt",
        "duplicate_per_gt",
        "median_delay",
        "p90_delay",
        "threshold",
        "base_recall",
        "min_recall",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def jsonable(value):
    """Convert common Python/NumPy/Path objects to JSON-serializable values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train candidate-level transition reliability head."
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/s2_reliability"))
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument(
        "--legality-gate",
        choices=["none", "cholec80", "train"],
        default="train",
    )
    parser.add_argument("--legal-split", default="train")
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--dwell-duration", type=int, default=5)
    parser.add_argument("--tolerance", type=int, default=30)
    parser.add_argument("--left-context", type=int, default=8)
    parser.add_argument(
        "--feature-set",
        choices=["none", "pa_pb", "prob5", "prob7"],
        default="pa_pb",
    )
    parser.add_argument(
        "--transition-identity",
        choices=["none", "phases", "edge"],
        default="none",
        help=(
            "Append transition identity features to every evidence bin: "
            "none, one-hot phase_from plus phase_to, or one-hot edge A->B."
        ),
    )
    parser.add_argument(
        "--num-phases",
        type=int,
        default=7,
        help="Number of phase classes for transition identity features.",
    )
    parser.add_argument(
        "--log-evidence",
        choices=["none", "instant", "accum", "both"],
        default="none",
        help=(
            "Append Liu-style transition-relative log evidence: instant "
            "log p(B)-log p(A), one-sided accumulated evidence, or both."
        ),
    )
    parser.add_argument(
        "--prototype-evidence",
        choices=["none", "phase_margin"],
        default="none",
        help=(
            "Append visual prototype evidence to every bin. phase_margin uses "
            "cos(z_t, proto_B) - cos(z_t, proto_A), with prototypes built from "
            "--prototype-split under --feature-root."
        ),
    )
    parser.add_argument(
        "--prototype-split",
        default="train",
        help="Feature split used to build visual phase prototypes.",
    )
    parser.add_argument(
        "--competing-phase-cues",
        action="store_true",
        help=(
            "Append transition-relative competing-phase cues to each bin: "
            "max p(other than A/B), B-vs-best-non-B margin, and B-vs-other margin."
        ),
    )
    parser.add_argument(
        "--prepost-prob-cues",
        action="store_true",
        help=(
            "Append candidate-level pre/post probability-distribution change cues "
            "(L1 and JS divergence) to each bin."
        ),
    )
    parser.add_argument(
        "--prepost-prob-cue-mode",
        choices=["both", "l1", "js"],
        default="both",
        help=(
            "Select which pre/post probability-distribution cue to use when "
            "--prepost-prob-cues is enabled."
        ),
    )
    parser.add_argument(
        "--visual-change-cue",
        action="store_true",
        help=(
            "Append candidate-level pre/post visual change magnitude from frozen "
            "features under --feature-root."
        ),
    )
    parser.add_argument("--encoder", choices=["tcn", "gru"], default="tcn")
    parser.add_argument(
        "--pooling",
        choices=["relative", "mean", "last"],
        default="relative",
        help=(
            "Temporal aggregation over the candidate window: learned relative "
            "evidence pooling, uniform mean pooling, or the last valid bin."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--ranking-weight", type=float, default=0.1)
    parser.add_argument("--ranking-margin", type=float, default=0.1)
    parser.add_argument(
        "--offset-weight",
        type=float,
        default=0.1,
        help="Weight lambda for the TriDet-style SmoothL1 boundary offset loss "
        "(0 disables the offset head/loss).",
    )
    parser.add_argument(
        "--offset-beta",
        type=float,
        default=1.0,
        help="SmoothL1 beta (frames) for the boundary offset loss.",
    )
    parser.add_argument("--recall-retention", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_legal_edges(args: argparse.Namespace) -> frozenset[tuple[int, int]] | None:
    if args.legality_gate == "none":
        return None
    if args.legality_gate == "cholec80":
        return CHOLEC80_WORKFLOW_EDGES
    if args.feature_root is None:
        raise SystemExit("--feature-root is required with --legality-gate=train.")
    return workflow_edges_from_feature_root(
        args.feature_root,
        split=args.legal_split,
        temporal_stride=args.temporal_stride,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        device = torch.device("cpu")

    legal_edges = resolve_legal_edges(args)
    phase_prototypes = None
    prototype_counts = None
    feature_dirs = {"train": None, "val": None, "test": None}
    needs_visual_features = args.prototype_evidence != "none" or args.visual_change_cue
    if needs_visual_features and args.feature_root is None:
        raise SystemExit(
            "--feature-root is required with --prototype-evidence or --visual-change-cue."
        )
    if args.prototype_evidence != "none":
        if args.feature_root is None:
            raise SystemExit("--feature-root is required with --prototype-evidence.")
        prototype_dir = resolve_feature_dir(args.feature_root, args.prototype_split)
        phase_prototypes, prototype_counts = build_phase_prototypes(prototype_dir)
    if needs_visual_features:
        feature_dirs["train"] = resolve_feature_dir(args.feature_root, args.train_dir.name)
        feature_dirs["val"] = resolve_feature_dir(args.feature_root, args.val_dir.name)
        if args.test_dir is not None:
            feature_dirs["test"] = resolve_feature_dir(args.feature_root, args.test_dir.name)

    train_dataset = TransitionCandidateDataset(
        args.train_dir,
        dwell_duration=args.dwell_duration,
        tolerance=args.tolerance,
        legal_edges=legal_edges,
        left_context=args.left_context,
        feature_set=args.feature_set,
        transition_identity=args.transition_identity,
        num_phases=args.num_phases,
        log_evidence=args.log_evidence,
        prototype_evidence=args.prototype_evidence,
        competing_phase_cues=args.competing_phase_cues,
        prepost_prob_cues=args.prepost_prob_cues,
        prepost_prob_cue_mode=args.prepost_prob_cue_mode,
        visual_change_cue=args.visual_change_cue,
        feature_dir=feature_dirs["train"],
        phase_prototypes=phase_prototypes,
    )
    val_dataset = TransitionCandidateDataset(
        args.val_dir,
        dwell_duration=args.dwell_duration,
        tolerance=args.tolerance,
        legal_edges=legal_edges,
        left_context=args.left_context,
        feature_set=args.feature_set,
        transition_identity=args.transition_identity,
        num_phases=args.num_phases,
        log_evidence=args.log_evidence,
        prototype_evidence=args.prototype_evidence,
        competing_phase_cues=args.competing_phase_cues,
        prepost_prob_cues=args.prepost_prob_cues,
        prepost_prob_cue_mode=args.prepost_prob_cue_mode,
        visual_change_cue=args.visual_change_cue,
        feature_dir=feature_dirs["val"],
        phase_prototypes=phase_prototypes,
    )
    test_dataset = None
    if args.test_dir is not None:
        test_dataset = TransitionCandidateDataset(
            args.test_dir,
            dwell_duration=args.dwell_duration,
            tolerance=args.tolerance,
            legal_edges=legal_edges,
            left_context=args.left_context,
            feature_set=args.feature_set,
            transition_identity=args.transition_identity,
            num_phases=args.num_phases,
            log_evidence=args.log_evidence,
            prototype_evidence=args.prototype_evidence,
            competing_phase_cues=args.competing_phase_cues,
            prepost_prob_cues=args.prepost_prob_cues,
            prepost_prob_cue_mode=args.prepost_prob_cue_mode,
            visual_change_cue=args.visual_change_cue,
            feature_dir=feature_dirs["test"],
            phase_prototypes=phase_prototypes,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_candidates,
        drop_last=False,
    )
    model = TransitionReliabilityHead(
        input_dim=feature_dim(
            args.feature_set,
            transition_identity=args.transition_identity,
            num_phases=args.num_phases,
            log_evidence=args.log_evidence,
            prototype_evidence=args.prototype_evidence,
            competing_phase_cues=args.competing_phase_cues,
            prepost_prob_cues=args.prepost_prob_cues,
            prepost_prob_cue_mode=args.prepost_prob_cue_mode,
            visual_change_cue=args.visual_change_cue,
        ),
        hidden_dim=args.hidden_dim,
        encoder_type=args.encoder,
        pooling_type=args.pooling,
        num_layers=args.num_layers,
        dropout=args.dropout,
        # Row 0 of the window is t_candidate - left_context + 1, so its offset
        # relative to t_candidate is -(left_context - 1). Enables the
        # TriDet-style expected boundary offset output and its SmoothL1 loss.
        window_offset_start=-(args.left_context - 1) if args.offset_weight > 0 else None,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(
        json.dumps(
            {
                "train_candidates": len(train_dataset),
                "train_positive": train_dataset.positive_count,
                "train_negative": train_dataset.negative_count,
                "val_candidates": len(val_dataset),
                "val_positive": val_dataset.positive_count,
                "val_negative": val_dataset.negative_count,
                "feature_set": args.feature_set,
                "transition_identity": args.transition_identity,
                "num_phases": args.num_phases,
                "log_evidence": args.log_evidence,
                "prototype_evidence": args.prototype_evidence,
                "prototype_split": args.prototype_split,
                "competing_phase_cues": args.competing_phase_cues,
                "prepost_prob_cues": args.prepost_prob_cues,
                "prepost_prob_cue_mode": args.prepost_prob_cue_mode,
                "visual_change_cue": args.visual_change_cue,
                "prototype_counts": (
                    {str(key): int(value) for key, value in sorted(prototype_counts.items())}
                    if prototype_counts is not None
                    else None
                ),
                "legal_edges": (
                    [list(edge) for edge in sorted(legal_edges)]
                    if legal_edges is not None
                    else None
                ),
            },
            indent=2,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_state = None
    best_summary = None
    best_score = -math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            ranking_weight=args.ranking_weight,
            ranking_margin=args.ranking_margin,
            offset_weight=args.offset_weight,
            offset_beta=args.offset_beta,
        )
        val_scores = predict_scores(model, val_dataset, device, args.batch_size)
        threshold, val_best, val_rows = sweep_thresholds(
            val_dataset,
            val_scores,
            tolerance=args.tolerance,
            recall_retention=args.recall_retention,
        )
        epoch_summary = {
            "epoch": epoch,
            "train": train_metrics,
            "val": compact_metrics(val_best),
        }
        history.append(epoch_summary)
        score = val_best["commit_f1"]
        if score > best_score:
            best_score = score
            best_state = {
                "model": model.state_dict(),
                "args": vars(args),
                "threshold": threshold,
                "val": compact_metrics(val_best),
                "val_sweep": [compact_metrics(row) for row in val_rows],
            }
            best_summary = epoch_summary
        print(json.dumps(epoch_summary))

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")
    torch.save(best_state, args.output_dir / "best_transition_reliability.pt")
    # Portable checkpoint for evaluation without training paths or optimizer state.
    portable_keys = (
        "dwell_duration", "tolerance", "left_context", "feature_set",
        "transition_identity", "num_phases", "log_evidence",
        "prototype_evidence", "competing_phase_cues", "prepost_prob_cues",
        "prepost_prob_cue_mode", "visual_change_cue", "encoder", "pooling",
        "hidden_dim", "num_layers", "dropout",
    )
    portable_checkpoint = args.output_dir / "tcv_model.pt"
    torch.save(
        {
            "model": best_state["model"],
            "config": {key: getattr(args, key) for key in portable_keys},
            "threshold": float(best_state["threshold"]),
            "seed": int(args.seed),
        },
        portable_checkpoint,
    )
    with (args.output_dir / "history.json").open("w") as file:
        json.dump(history, file, indent=2)

    model.load_state_dict(best_state["model"])
    val_scores = predict_scores(model, val_dataset, device, args.batch_size)
    export_trigger_table(
        val_dataset,
        val_scores,
        threshold=best_state["threshold"],
        tolerance=args.tolerance,
        output_path=args.output_dir / "val_trigger_table.csv",
        seed=args.seed,
        dwell_duration=args.dwell_duration,
        legal_edges=legal_edges,
    )
    final_summary = {
        "best": {
            key: jsonable(value)
            for key, value in best_state.items()
            if key != "model"
        },
        "best_epoch_summary": jsonable(best_summary),
        "checkpoint": str((args.output_dir / "best_transition_reliability.pt").resolve()),
        "portable_checkpoint": str(portable_checkpoint.resolve()),
        "val_trigger_table": str((args.output_dir / "val_trigger_table.csv").resolve()),
    }
    if test_dataset is not None:
        test_scores = predict_scores(model, test_dataset, device, args.batch_size)
        test_metrics = evaluate_scores(
            test_dataset,
            test_scores,
            threshold=best_state["threshold"],
            tolerance=args.tolerance,
        )
        final_summary["test"] = compact_metrics(test_metrics)
        final_summary["msp_test_at_0_60"] = compact_metrics(
            evaluate_scores(
                test_dataset,
                test_dataset.msp_scores,
                threshold=0.60,
                tolerance=args.tolerance,
            )
        )
        export_trigger_table(
            test_dataset,
            test_scores,
            threshold=best_state["threshold"],
            tolerance=args.tolerance,
            output_path=args.output_dir / "test_trigger_table.csv",
            seed=args.seed,
            dwell_duration=args.dwell_duration,
            legal_edges=legal_edges,
        )
        final_summary["test_trigger_table"] = str(
            (args.output_dir / "test_trigger_table.csv").resolve()
        )

    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(final_summary, file, indent=2)
    print(json.dumps(final_summary, indent=2, default=str))


if __name__ == "__main__":
    main()
