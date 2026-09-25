"""Evaluate a released TCV checkpoint on aligned recognizer trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from event_commit_metrics import workflow_edges_from_feature_root
from models.transition_reliability import TransitionReliabilityHead
from train_transition_reliability import (
    TransitionCandidateDataset,
    compact_metrics,
    evaluate_scores,
    export_trigger_table,
    feature_dim,
    predict_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    legal_edges = workflow_edges_from_feature_root(args.feature_root, split="train")
    feature_dir = args.feature_root / args.trajectory_dir.name
    dataset = TransitionCandidateDataset(
        args.trajectory_dir,
        dwell_duration=cfg["dwell_duration"],
        tolerance=cfg["tolerance"],
        legal_edges=legal_edges,
        left_context=cfg["left_context"],
        feature_set=cfg["feature_set"],
        transition_identity=cfg["transition_identity"],
        num_phases=cfg["num_phases"],
        log_evidence=cfg["log_evidence"],
        prototype_evidence=cfg["prototype_evidence"],
        competing_phase_cues=cfg["competing_phase_cues"],
        prepost_prob_cues=cfg["prepost_prob_cues"],
        prepost_prob_cue_mode=cfg["prepost_prob_cue_mode"],
        visual_change_cue=cfg["visual_change_cue"],
        feature_dir=feature_dir,
    )
    model = TransitionReliabilityHead(
        input_dim=feature_dim(
            cfg["feature_set"],
            transition_identity=cfg["transition_identity"],
            num_phases=cfg["num_phases"],
            log_evidence=cfg["log_evidence"],
            prototype_evidence=cfg["prototype_evidence"],
            competing_phase_cues=cfg["competing_phase_cues"],
            prepost_prob_cues=cfg["prepost_prob_cues"],
            prepost_prob_cue_mode=cfg["prepost_prob_cue_mode"],
            visual_change_cue=cfg["visual_change_cue"],
        ),
        hidden_dim=cfg["hidden_dim"],
        encoder_type=cfg["encoder"],
        pooling_type=cfg["pooling"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device).eval()
    scores = predict_scores(model, dataset, device, batch_size=128)
    metrics = compact_metrics(evaluate_scores(dataset, scores, checkpoint["threshold"], cfg["tolerance"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)
    export_trigger_table(
        dataset,
        scores,
        threshold=checkpoint["threshold"],
        tolerance=cfg["tolerance"],
        output_path=args.output_dir / "trigger_table.csv",
        seed=checkpoint["seed"],
        dwell_duration=cfg["dwell_duration"],
        legal_edges=legal_edges,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
