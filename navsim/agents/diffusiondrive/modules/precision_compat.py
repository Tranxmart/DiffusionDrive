"""Shared helpers for mixed-precision compatibility.

`F.interpolate(..., mode="bilinear")` (and `nn.Upsample` with bilinear mode,
which calls it) is not implemented for BFloat16 on some torch builds
(e.g. torch 2.x + CUDA 12 on A100). bf16-mixed training therefore crashes
with:

    RuntimeError: "upsample_bilinear2d_out_frame" not implemented for 'BFloat16'

The `bilinear_resize` helper transparently casts bf16 inputs to fp32, runs
the bilinear interpolation, and casts back, so call sites work unchanged
under fp32, fp16-mixed and bf16-mixed.
"""

from typing import Tuple

import torch
import torch.nn.functional as F


def bilinear_resize(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Bilinear-interpolate to `size`, working around missing bf16 kernels.

    :param x: input tensor (N, C, H, W); any dtype
    :param size: target (H, W)
    :return: resized tensor, same dtype as input
    """
    if x.dtype == torch.bfloat16:
        return F.interpolate(x.float(), size=size, mode="bilinear", align_corners=False).to(
            torch.bfloat16
        )
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class BilinearUpsample(torch.nn.Module):
    """nn.Upsample replacement (bilinear, align_corners=False) with bf16 support.

    Supports either `scale_factor` or `size` (mutually exclusive), matching
    the subset of nn.Upsample options used in this codebase.
    """

    def __init__(
        self,
        size: Tuple[int, int] = None,
        scale_factor: float = None,
    ) -> None:
        super().__init__()
        assert (size is None) != (scale_factor is None), "exactly one of size / scale_factor"
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.size is not None:
            return bilinear_resize(x, self.size)
        out_h = int(x.shape[2] * self.scale_factor)
        out_w = int(x.shape[3] * self.scale_factor)
        return bilinear_resize(x, (out_h, out_w))