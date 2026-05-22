"""On-the-fly degradation dataset for multispectral satellite TIFFs.

Generates LQ from GT inside __getitem__ using only channel-agnostic
degradations: gaussian blur, downscale, gaussian noise. JPEG/sinc/USM
and any RGB-perceptual degradations are deliberately omitted — they
model 8-bit photography artifacts, not satellite radiometry.

The (lq, gt) pairs returned look identical to those from
`paired_multispectral`, so this dataset works with the standard `image`
model_type without touching any model code.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils import data
from torchvision.transforms.functional import normalize

from neosr.data.transforms import basic_augment
from neosr.utils import get_root_logger, scandir, tc
from neosr.utils.multispectral_io import read_tiff
from neosr.utils.registry import DATASET_REGISTRY

if TYPE_CHECKING:
    pass

TIFF_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")


def _gaussian_kernel1d(sigma: float, radius: int) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _apply_gaussian_blur(img: Tensor, sigma: float) -> Tensor:
    """img: (C, H, W). Applies separable gaussian blur per-channel."""
    if sigma <= 1e-6:
        return img
    radius = max(1, int(3.0 * sigma))
    kernel1d = _gaussian_kernel1d(sigma, radius)
    c = img.shape[0]
    kx = kernel1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = kernel1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    img = img.unsqueeze(0)
    img = F.conv2d(img, kx, padding=(0, radius), groups=c)
    img = F.conv2d(img, ky, padding=(radius, 0), groups=c)
    return img.squeeze(0)


@DATASET_REGISTRY.register()
class otf_multispectral(data.Dataset):
    """OTF degradation dataset for multispectral TIFFs.

    Required opt keys: dataroot_gt, scale, phase.
    Optional opt keys: patch_size, use_hflip, use_rot,
        blur_sigma_range, noise_sigma_range, downscale_mode, mean, std.
    """

    def __init__(self, opt: dict[str, Any]) -> None:
        super().__init__()
        self.opt = opt
        self.mean = opt.get("mean")
        self.std = opt.get("std")
        self.gt_folder = opt["dataroot_gt"]
        self.scale = int(opt["scale"])
        self.blur_sigma_range = tuple(opt.get("blur_sigma_range", (0.2, 1.5)))
        self.noise_sigma_range = tuple(opt.get("noise_sigma_range", (0.0, 0.02)))
        self.downscale_mode = opt.get("downscale_mode", "bicubic")

        self.paths = sorted(scandir(self.gt_folder, suffix=TIFF_SUFFIXES, full_path=True))
        if not self.paths:
            msg = f"No TIFF files found in {self.gt_folder}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.paths)

    def _degrade(self, gt: Tensor) -> Tensor:
        """gt: (C, H_gt, W_gt) float32 in [0, 1]. Returns lq (C, H_lq, W_lq)."""
        blur_sigma = float(np.random.uniform(*self.blur_sigma_range))
        lq = _apply_gaussian_blur(gt, blur_sigma)

        lq = F.interpolate(
            lq.unsqueeze(0),
            scale_factor=1.0 / self.scale,
            mode=self.downscale_mode,
            antialias=self.downscale_mode in {"bicubic", "bilinear"},
        ).squeeze(0)

        noise_sigma = float(np.random.uniform(*self.noise_sigma_range))
        if noise_sigma > 0:
            lq = lq + torch.randn_like(lq) * noise_sigma

        return lq.clamp_(0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        logger = get_root_logger()
        gt_path = self.paths[index]

        try:
            img_gt = read_tiff(gt_path)
        except Exception as e:  # noqa: BLE001
            msg = f"{tc.red}Failed to read TIFF {gt_path}: {e}{tc.end}"
            logger.error(msg)
            sys.exit(1)

        # Crop GT to a multiple of scale so LQ has integer size.
        if self.opt["phase"] == "train":
            patch_size = int(self.opt["patch_size"])
            gt_patch = patch_size * self.scale
            h, w = img_gt.shape[:2]
            if h < gt_patch or w < gt_patch:
                msg = (
                    f"Image {gt_path} ({h}x{w}) smaller than required GT patch "
                    f"{gt_patch}x{gt_patch}"
                )
                raise ValueError(msg)
            top = np.random.randint(0, h - gt_patch + 1)
            left = np.random.randint(0, w - gt_patch + 1)
            img_gt = img_gt[top : top + gt_patch, left : left + gt_patch, :]
            img_gt = basic_augment(  # type: ignore[assignment]
                img_gt,
                hflip=self.opt.get("use_hflip", True),
                rotation=self.opt.get("use_rot", True),
            )
        else:
            h, w = img_gt.shape[:2]
            img_gt = img_gt[: (h // self.scale) * self.scale, : (w // self.scale) * self.scale, :]

        tensor_gt = torch.from_numpy(img_gt.transpose(2, 0, 1).copy()).float()
        tensor_lq = self._degrade(tensor_gt)

        if self.mean is not None or self.std is not None:
            normalize(tensor_lq, self.mean, self.std, inplace=True)
            normalize(tensor_gt, self.mean, self.std, inplace=True)

        return {
            "lq": tensor_lq,
            "gt": tensor_gt,
            "lq_path": gt_path,
            "gt_path": gt_path,
        }
