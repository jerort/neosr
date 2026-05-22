"""RGB-subset wrapper (Strategy A) for adapting RGB-locked losses to
multispectral inputs.

Strategy A — pass only the user-specified RGB-equivalent bands through
the inner loss. Used to wrap losses whose backbone is a pretrained net
that expects 3-channel input (VGG19, DINOv2, etc.).

Strategy B (per-triplet sliding) is deliberately NOT implemented here.
Feeding out-of-distribution band combos (e.g. NIR-NIR-NIR) through
RGB-pretrained nets produces feature activations whose meaning for SR
quality is unclear — it pretends to supervise all bands but actually
regularises toward "looks like a natural photo." Strategy A is more
principled even though it only supervises 3 bands directly.

Losses with no pretrained weights (gan disc, ssim, ldl, L1, Huber,
wavelet) should NOT be wrapped — they handle arbitrary channel counts
natively (Strategy C). Just set the channel count in the config.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class rgb_subset_loss(nn.Module):
    """Wraps another loss and applies it only to the bands at ``rgb_indices``.

    Args:
    ----
        inner (dict): Inner loss spec. Must contain ``type`` (registered
            loss name) and any kwargs for that loss.
        rgb_indices (list[int]): Indices of the 3 bands to pass to the inner
            loss, in [R, G, B] order. Must have length 3.
        loss_weight (float): Outer weight applied on top of the inner loss.
            Default: 1.0.
    """

    def __init__(
        self,
        inner: dict[str, Any],
        rgb_indices: list[int],
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if len(rgb_indices) != 3:
            msg = f"rgb_indices must have length 3, got {rgb_indices}"
            raise ValueError(msg)

        inner_opts = dict(inner)
        inner_type = inner_opts.pop("type")
        inner_cls = LOSS_REGISTRY.get(inner_type)
        if inner_cls is None:
            msg = f"Unknown inner loss type: {inner_type}"
            raise ValueError(msg)
        self.inner: nn.Module = inner_cls(**inner_opts)  # type: ignore[operator]

        self.register_buffer(
            "rgb_indices_t", torch.tensor(rgb_indices, dtype=torch.long), persistent=False
        )
        self.loss_weight = loss_weight

    def forward(self, pred: Tensor, target: Tensor, **kwargs: Any) -> Tensor:
        pred_rgb = pred.index_select(1, self.rgb_indices_t)
        target_rgb = target.index_select(1, self.rgb_indices_t)
        return self.loss_weight * self.inner(pred_rgb, target_rgb, **kwargs)
