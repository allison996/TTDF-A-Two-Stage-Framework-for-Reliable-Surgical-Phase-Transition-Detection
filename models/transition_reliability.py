"""Candidate-level transition reliability modules.

This module is for the S2 stage of the phase-recognition project.  It does not
replace the frame-wise recognizer.  Instead, it consumes a short window of
recognizer evidence around a proposed transition A->B and predicts whether that
candidate should be committed.

The design borrows the useful part of TriDet's Trident-head: local boundaries
are represented by a relative distribution over neighboring temporal bins,
rather than by a single point confidence.  Here that idea becomes relative
transition-evidence pooling over a candidate window.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvBlock(nn.Module):
    """Small causal 1D conv block for short transition windows."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive.")
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T].  Padding is only on the left, so no future bin leaks in.
        residual = x
        x = F.pad(x, (self.left_padding, 0))
        x = self.conv(x)
        x = F.relu(self.norm(x))
        x = self.dropout(x)
        return x + residual


#窗口相对证据分布
# α(τ) = relative transition evidence distribution
# o_c = E[τ] = Σ α(τ) τ
# r_c = P(commit-correct)
class RelativeEvidencePooling(nn.Module):
    """TriDet-style relative evidence distribution over a short window.

    Given encoded window features H in shape [B, C, T], this layer predicts one
    scalar evidence logit per relative temporal bin, normalizes the logits with a
    softmax over T, and returns a weighted candidate representation.
    """

    def __init__(self, channels: int, init_std: float = 0.01):
        super().__init__()
        self.evidence_head = nn.Conv1d(channels, 1, kernel_size=1)
        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        # TriDet initializes boundary heads with a small Gaussian for stability.
        # Here this makes the initial relative evidence distribution close to
        # uniform, then training learns which relative bins matter.
        nn.init.normal_(self.evidence_head.weight, mean=0.0, std=init_std)
        nn.init.constant_(self.evidence_head.bias, 0.0)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, C, T], got {tuple(x.shape)}.")
        logits = self.evidence_head(x).squeeze(1)  # [B, T]
        if mask is not None:
            if mask.shape != logits.shape:
                raise ValueError(
                    f"mask must have shape {tuple(logits.shape)}, got {tuple(mask.shape)}."
                )
            logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(1), dim=-1)
        return pooled, weights


class TransitionReliabilityHead(nn.Module):
    """Short-window transition verifier.

    Parameters
    ----------
    input_dim:
        Number of per-bin evidence channels.  A minimal S2 model can use only
        [p_A, p_B] and set input_dim=2.  A richer version may use
        [p_A, p_B, max_prob, entropy, margin, delta_p_B, delta_entropy].
    hidden_dim:
        Channel width for the short-window encoder.
    encoder_type:
        "tcn" for a lightweight causal convolutional encoder, or "gru" for a
        recurrent encoder.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        encoder_type: str = "tcn",
        pooling_type: str = "relative",
        num_layers: int = 1,
        kernel_size: int = 3,
        dropout: float = 0.1,
        evidence_init_std: float = 0.01,
        window_offset_start: int | None = None,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        if encoder_type not in {"tcn", "gru"}:
            raise ValueError("encoder_type must be 'tcn' or 'gru'.")
        if pooling_type not in {"relative", "mean", "last"}:
            raise ValueError("pooling_type must be 'relative', 'mean', or 'last'.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.pooling_type = pooling_type
        # Frame offset (relative to t_candidate) of window row 0, e.g.
        # -(left_context - 1). When set, forward() also returns a TriDet-style
        # expected boundary offset o_c = sum_tau alpha(tau) * tau, reusing the
        # same relative evidence distribution alpha used for pooling.
        self.window_offset_start = window_offset_start

        if encoder_type == "tcn":
            self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
            self.encoder = nn.Sequential(
                *[
                    CausalConvBlock(
                        hidden_dim, kernel_size=kernel_size, dropout=dropout
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.encoder = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )

        self.pool = RelativeEvidencePooling(hidden_dim, init_std=evidence_init_std)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, window: torch.Tensor, mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Predict candidate commit reliability.

        Parameters
        ----------
        window:
            Tensor with shape [B, T, D].  T is the relative candidate window and
            D is the evidence dimension.
        mask:
            Optional boolean tensor [B, T] for padded candidate windows.

        Returns
        -------
        A dict containing:
        - logit: unnormalized reliability score [B]
        - probability: sigmoid(logit), i.e. P(commit-correct)
        - evidence_weights: relative evidence distribution alpha(tau) over T [B, T]
        - offset: expected boundary offset o_c = sum_tau alpha(tau) * tau, in
          frames relative to t_candidate [B].  Only present when
          window_offset_start was set at construction time.
        """

        if window.ndim != 3:
            raise ValueError(
                f"window must have shape [B, T, D], got {tuple(window.shape)}."
            )
        if window.shape[-1] != self.input_dim:
            raise ValueError(
                f"window feature dimension must be {self.input_dim}, "
                f"got {window.shape[-1]}."
            )

        if self.encoder_type == "tcn":
            x = window.transpose(1, 2)  # [B, D, T]
            x = self.input_proj(x)
            x = self.encoder(x)
        else:
            x, _ = self.encoder(window)
            x = x.transpose(1, 2)  # [B, C, T]

        if self.pooling_type == "relative":
            pooled, evidence_weights = self.pool(x, mask=mask)
        elif self.pooling_type == "mean":
            if mask is None:
                evidence_weights = torch.full(
                    (x.shape[0], x.shape[-1]),
                    1.0 / max(x.shape[-1], 1),
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                valid = mask.bool().to(device=x.device)
                counts = valid.sum(dim=-1, keepdim=True).clamp_min(1)
                evidence_weights = valid.to(dtype=x.dtype) / counts.to(dtype=x.dtype)
            pooled = torch.sum(x * evidence_weights.unsqueeze(1), dim=-1)
        else:
            if mask is None:
                last_indices = torch.full(
                    (x.shape[0],),
                    x.shape[-1] - 1,
                    device=x.device,
                    dtype=torch.long,
                )
            else:
                valid = mask.bool().to(device=x.device)
                lengths = valid.long().sum(dim=-1).clamp_min(1)
                last_indices = lengths - 1
            evidence_weights = torch.zeros(
                (x.shape[0], x.shape[-1]), device=x.device, dtype=x.dtype
            )
            evidence_weights.scatter_(1, last_indices.unsqueeze(1), 1.0)
            pooled = x.gather(
                2, last_indices.view(-1, 1, 1).expand(-1, x.shape[1], 1)
            ).squeeze(-1)
        logit = self.classifier(pooled).squeeze(-1)
        output = {
            "logit": logit,
            "probability": torch.sigmoid(logit),
            "evidence_weights": evidence_weights,
        }

        if self.window_offset_start is not None:
            # TriDet decodes a boundary offset as E_{b~p}[b], the expectation
            # of the relative bin index under the predicted relative
            # probability distribution.  We reuse alpha(tau) = evidence_weights
            # the same way, over the window's frame offsets relative to
            # t_candidate instead of bin indices.
            length = evidence_weights.shape[-1]
            positions = torch.arange(
                self.window_offset_start,
                self.window_offset_start + length,
                device=evidence_weights.device,
                dtype=evidence_weights.dtype,
            )
            output["offset"] = torch.sum(evidence_weights * positions, dim=-1)

        return output


def sigmoid_focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal BCE for commit-correctness prediction."""

    logits = logits.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError("reduction must be one of 'none', 'mean', or 'sum'.")


def correctness_ranking_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Pairwise ranking loss for transition reliability scores.

    Encourages true commits to receive higher reliability scores than false
    commits in the same mini-batch.  This is a lightweight candidate-level
    analogue of correctness ranking losses used for confidence learning.
    """

    scores = scores.float()
    targets = targets.bool()
    positive_scores = scores[targets]
    negative_scores = scores[~targets]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return scores.new_tensor(0.0)
    violations = margin - positive_scores[:, None] + negative_scores[None, :]
    return F.relu(violations).mean()


def smooth_l1_offset_loss(
    predicted_offset: torch.Tensor,
    target_offset: torch.Tensor,
    positive_mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """SmoothL1 loss between predicted and ground-truth boundary offset.

    Mirrors TriDet's boundary regression: only candidates matched to a
    ground-truth transition (positive_mask) have a defined target offset, so
    negatives are excluded rather than being pulled toward zero.
    """

    predicted_offset = predicted_offset.float()
    target_offset = target_offset.float()
    positive_mask = positive_mask.bool()
    if positive_mask.sum() == 0:
        return predicted_offset.new_tensor(0.0)
    return F.smooth_l1_loss(
        predicted_offset[positive_mask],
        target_offset[positive_mask],
        beta=beta,
    )
