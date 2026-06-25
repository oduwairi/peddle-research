"""4-head regression model on top of a HuggingFace encoder backbone.

The backbone is anything ``AutoModel`` can load (the plan picks
``microsoft/deberta-v3-base``); the regression head is a tiny MLP that maps
the pooled ``[CLS]`` representation to four scalars — ``composite``,
``survivability``, ``engagement_volume``, ``engagement_velocity`` (in that
order; see :data:`draper.scoring_predictor.data.HEAD_NAMES`).

Loss is masked, sample-weighted MSE summed across the four heads:

* For each head, MSE is computed only over rows where ``target_mask`` is
  ``True`` (Reddit + ``other`` rows mask the engagement heads — see
  :func:`draper.scoring_predictor.data._target_mask_for_platform`).
* Each row's loss is multiplied by its ``sample_weight`` (derived from the
  ``training_quality`` rating).
* Each head is multiplied by a config-supplied ``head_weight`` so we can
  rebalance heads without retraining.

The HF ``Trainer`` calls ``forward(**batch)`` and reads ``loss`` /
``logits`` from the returned dict; this signature matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — typing only
    import torch
    from torch import nn

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from draper.scoring_predictor.data import HEAD_NAMES

NUM_HEADS = len(HEAD_NAMES)


@dataclass
class RegressorOutput:
    """Loss + per-head predictions for one batch."""

    loss: torch.Tensor
    logits: torch.Tensor  # shape: (batch, NUM_HEADS)


class FourHeadRegressor(nn.Module):
    """Encoder + small MLP regression head, four scalar outputs."""

    def __init__(
        self,
        *,
        model_name: str,
        dropout: float = 0.1,
        head_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        backbone_config = AutoConfig.from_pretrained(model_name)
        # Force fp32 explicitly. `microsoft/deberta-v3-base` has
        # `torch_dtype=float16` in its HF config, so without this override
        # `from_pretrained` silently loads the backbone in fp16. That dtype
        # mismatch with our fp32 head causes a `mat1 and mat2 must have the
        # same dtype` error in fp32 mode, and worse — silently produces NaN
        # gradients under bf16/fp16 mixed precision because fp16 underflows
        # on the small intermediate values in DeBERTa-v3's disentangled
        # attention. This was the actual root cause of the NaN explosions
        # in attempts 4–6, mistakenly attributed to learning rate / head
        # init / mean-pooling. Trainer's `bf16=True` flag will autocast on
        # top of an fp32 backbone correctly when we want bf16 again.
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        hidden_size = int(getattr(backbone_config, "hidden_size", 768))
        # Single linear head — matches HF's `DebertaV2ForSequenceClassification`
        # standard. The previous Linear→GELU→Dropout→Linear MLP added 590k
        # randomly-initialized parameters that competed with the backbone for
        # gradient signal at the same LR, contributing to "predict-the-mean"
        # collapse. A single linear is faster to learn from scratch and the
        # backbone already provides the non-linearity needed for regression.
        self.head = nn.Linear(hidden_size, NUM_HEADS)
        # Small-std weight init for the head. Pretrained DeBERTa-v3 hidden
        # states have large magnitude (typical norm ~15 per pooled vector),
        # and PyTorch's default Kaiming-uniform init on `nn.Linear(768, 4)`
        # gives output magnitude ~9, which combined with our higher head LR
        # blows up MSE on the first gradient step (verified empirically:
        # loss=7927 at step 50 with default init + lr=1e-3). std=0.02 keeps
        # initial outputs near zero so MSE starts in a sane range and
        # gradients stay bounded.
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        # Register head_weights as a non-trainable buffer so the values follow
        # the model to whatever device it's moved to.
        self.register_buffer(
            "head_weights",
            torch.tensor(head_weights, dtype=torch.float32),
            persistent=False,
        )
        # Annotate for mypy (register_buffer doesn't preserve type info)
        self.head_weights: torch.Tensor
        # Sigmoid-bounded output. Earlier attempts attributed slow learning
        # to sigmoid+MSE gradient saturation, but the root cause turned out
        # to be (a) raw `last_hidden_state[:, 0, :]` pooling on DeBERTa-v3
        # (no pooler training) and (b) head learning at same LR as backbone
        # (predict-the-mean collapse). With mean-pooling and 50x layer-wise
        # LR for the head, sigmoid is the right choice: it bounds output to
        # [0,1] matching the target range, prevents the MSE-on-linear blow-up
        # we hit at head_lr=1e-3 (loss=7927 at step 50 → NaN gradients), and
        # is the standard for bounded regression in HF.

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        """Forward HF Trainer's gradient-checkpointing flag to the backbone.

        ``Trainer`` calls this on the top-level model, but our backbone is
        nested under ``self.backbone`` so without this passthrough the flag
        silently no-ops (keeping all activations in memory).
        """
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Forward HF Trainer's gradient-checkpointing flag to the backbone."""
        self.backbone.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        encoder_kwargs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**encoder_kwargs)
        # Mean-pool over non-padding tokens. DeBERTa-v3's raw `last_hidden_state[:, 0, :]`
        # is NOT pooler-trained (the `ContextPooler` lives in
        # `DebertaV2ForSequenceClassification`, not in `AutoModel`), so directly
        # using the CLS embedding produces a noisy regression anchor — empirically
        # this caused predict-the-mean collapse on this task. Mean-pooling over
        # non-pad positions is the Kaggle-winner standard for DeBERTa regression
        # (Feedback Prize 2022, AES 2.0 2024).
        last_hidden = outputs.last_hidden_state  # (B, T, H)
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        pooled = self.dropout(pooled)
        # Sigmoid-bounded output. See class-level docstring for rationale.
        raw = self.head(pooled)
        logits = torch.sigmoid(raw)

        result: dict[str, torch.Tensor] = {"logits": logits}

        if targets is not None:
            assert target_mask is not None, "target_mask required when targets are passed"
            assert sample_weight is not None, "sample_weight required when targets are passed"
            result["loss"] = _masked_weighted_mse(
                logits,
                targets=targets,
                target_mask=target_mask,
                sample_weight=sample_weight,
                head_weights=self.head_weights,
            )
        return result


def _masked_weighted_mse(
    predictions: torch.Tensor,
    *,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    head_weights: torch.Tensor,
) -> torch.Tensor:
    """Per-element MSE, masked + weighted, averaged over valid elements.

    Shapes:
        predictions:   (B, H)  float
        targets:       (B, H)  float
        target_mask:   (B, H)  bool / float
        sample_weight: (B,)    float
        head_weights:  (H,)    float

    Returns a scalar. We average over the count of valid (mask=True) elements
    rather than batch-size so a batch heavy with weak-platform rows (which
    mask 2/4 heads) doesn't get artificially low loss.

    Raises ValueError if the batch has no valid (mask=True) elements — this
    indicates a data issue that should surface loudly, not be hidden by a
    fallback loss of 0.0.
    """
    mask = target_mask.float()
    sq_err = (predictions - targets) ** 2  # (B, H)
    # Per-element weight = sample_weight (broadcast across heads) * head_weights
    weight = sample_weight.unsqueeze(1) * head_weights.unsqueeze(0)  # (B, H)
    weighted = sq_err * weight * mask
    denom = (weight * mask).sum()
    if denom <= 0.0:
        raise ValueError(
            f"No valid (masked) elements in batch: all {predictions.shape[0]} "
            f"rows × {predictions.shape[1]} heads are masked out or zero-weighted. "
            "This indicates a data problem (e.g., a batch of only quality-1 "
            "or weak-platform rows)."
        )
    return weighted.sum() / denom


def predictions_to_dict(
    predictions: torch.Tensor,
) -> list[dict[str, float]]:
    """Convert a ``(batch, NUM_HEADS)`` prediction tensor to per-row dicts."""
    rows = predictions.detach().cpu().tolist()
    return [
        {name: float(val) for name, val in zip(HEAD_NAMES, row, strict=True)}
        for row in rows
    ]
