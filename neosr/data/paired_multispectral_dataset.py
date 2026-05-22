"""Paired multispectral TIFF dataset for satellite imagery SR.

Mirrors `paired` but bypasses the 8-bit RGB assumptions baked into
imfrombytes (cv2.IMREAD_COLOR + /255) and img2tensor (BGR↔RGB swap).
TIFFs are loaded at full 16-bit depth via tifffile, normalised /65535,
and stay in their native band order — no color-space conversions.

The band count is implicit in the data: whatever bands are in the file
get passed through. The training config's num_in_ch / num_out_ch must
match. Band selection is a preprocessing step outside this loader.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.utils import data
from torchvision.transforms.functional import normalize

from neosr.data.transforms import basic_augment, paired_random_crop
from neosr.utils import get_root_logger, scandir, tc
from neosr.utils.multispectral_io import read_tiff
from neosr.utils.registry import DATASET_REGISTRY

if TYPE_CHECKING:
    import numpy as np

TIFF_SUFFIXES = (".tif", ".tiff", ".TIF", ".TIFF")


def _paired_tiff_paths(gt_folder: str, lq_folder: str) -> list[dict[str, str]]:
    gt_paths = sorted(scandir(gt_folder, suffix=TIFF_SUFFIXES, full_path=True))
    lq_paths = sorted(scandir(lq_folder, suffix=TIFF_SUFFIXES, full_path=True))
    if len(gt_paths) != len(lq_paths):
        msg = (
            f"GT and LQ folders have different counts: "
            f"{len(gt_paths)} vs {len(lq_paths)}"
        )
        raise ValueError(msg)
    if not gt_paths:
        msg = f"No TIFF files found in {gt_folder}"
        raise ValueError(msg)
    # Pair by filename stem (sorted order is unreliable when names differ).
    gt_by_stem = {Path(p).stem: p for p in gt_paths}
    lq_by_stem = {Path(p).stem: p for p in lq_paths}
    common = sorted(set(gt_by_stem) & set(lq_by_stem))
    if len(common) != len(gt_paths):
        missing = set(gt_by_stem) ^ set(lq_by_stem)
        msg = f"GT/LQ stem mismatch; unmatched: {sorted(missing)[:5]} ..."
        raise ValueError(msg)
    return [{"gt_path": gt_by_stem[s], "lq_path": lq_by_stem[s]} for s in common]


@DATASET_REGISTRY.register()
class paired_multispectral(data.Dataset):
    """Paired multispectral TIFF dataset for SR training.

    Required opt keys: dataroot_gt, dataroot_lq, scale, phase.
    Optional opt keys: patch_size, use_hflip, use_rot, mean, std.
    """

    def __init__(self, opt: dict[str, Any]) -> None:
        super().__init__()
        self.opt = opt
        self.mean = opt.get("mean")
        self.std = opt.get("std")
        self.gt_folder = opt["dataroot_gt"]
        self.lq_folder = opt["dataroot_lq"]
        self.paths: list[dict[str, str]] = _paired_tiff_paths(
            self.gt_folder, self.lq_folder
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, str | Tensor]:
        logger = get_root_logger()
        gt_path = self.paths[index]["gt_path"]
        lq_path = self.paths[index]["lq_path"]

        try:
            img_gt: np.ndarray = read_tiff(gt_path)
            img_lq: np.ndarray = read_tiff(lq_path)
        except Exception as e:  # noqa: BLE001
            msg = f"{tc.red}Failed to read TIFF pair {gt_path} / {lq_path}: {e}{tc.end}"
            logger.error(msg)
            sys.exit(1)

        if img_gt.shape[-1] != img_lq.shape[-1]:
            msg = (
                f"GT and LQ band counts differ ({img_gt.shape[-1]} vs "
                f"{img_lq.shape[-1]}): {gt_path}"
            )
            raise ValueError(msg)

        scale = self.opt["scale"]

        if self.opt["phase"] == "train":
            patch_size = self.opt["patch_size"]
            flip = self.opt.get("use_hflip", True)
            rot = self.opt.get("use_rot", True)
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, patch_size, scale, gt_path
            )
            img_gt, img_lq = basic_augment(  # type: ignore[misc,assignment]
                [img_gt, img_lq], hflip=flip, rotation=rot
            )
        else:
            img_gt = img_gt[0 : img_lq.shape[0] * scale, 0 : img_lq.shape[1] * scale, :]

        # HWC → CHW, numpy → tensor, NO bgr2rgb swap.
        tensor_gt = torch.from_numpy(img_gt.transpose(2, 0, 1).copy()).float()
        tensor_lq = torch.from_numpy(img_lq.transpose(2, 0, 1).copy()).float()

        if self.mean is not None or self.std is not None:
            normalize(tensor_lq, self.mean, self.std, inplace=True)
            normalize(tensor_gt, self.mean, self.std, inplace=True)

        return {
            "lq": tensor_lq,
            "gt": tensor_gt,
            "lq_path": lq_path,
            "gt_path": gt_path,
        }
