"""Multispectral TIFF IO for neosr satellite-imagery fork.

Bypasses cv2.imdecode (which forces 3-ch BGR uint8) and returns float32
arrays normalized to [0, 1] via /65535.0, preserving full 16-bit depth.

Band selection is treated as a preprocessing step outside training: this
loader returns whatever bands are present in the file. The model's
num_in_ch / num_out_ch in the training config must match.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tifffile

UINT16_MAX = 65535.0


def read_tiff(path: str | Path) -> np.ndarray:
    """Load a TIFF as float32 (H, W, C) in [0, 1].

    Single-band TIFFs are returned with an explicit channel axis so the rest
    of the pipeline can assume HWC throughout.
    """
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 16:
        # tifffile may return (C, H, W) for planar configs; normalise to HWC
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32) / UINT16_MAX


def read_tiff_bytes(content: bytes) -> np.ndarray:
    """Same as read_tiff but from an in-memory bytes buffer (lmdb path)."""
    import io

    arr = tifffile.imread(io.BytesIO(content))
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 16:
        arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32) / UINT16_MAX


def write_tiff(path: str | Path, arr: np.ndarray) -> None:
    """Inverse of read_tiff: take a float (H, W, C) array in [0, 1] and write a
    uint16 TIFF, preserving channel count and 16-bit depth.

    Values are clipped to [0, 1] and scaled by 65535. A single-channel array may
    be (H, W) or (H, W, 1); the trailing axis is squeezed so it writes as a plain
    grayscale TIFF. Geospatial metadata (CRS/transform) is NOT carried — this only
    preserves pixel values, bands, and bit depth.
    """
    arr = np.clip(arr, 0.0, 1.0)
    out = np.rint(arr * UINT16_MAX).astype(np.uint16)
    if out.ndim == 3 and out.shape[-1] == 1:
        out = out[..., 0]
    tifffile.imwrite(str(path), out)


def write_tiff_dn(path: str | Path, arr: np.ndarray, ceilings: Sequence[float]) -> None:
    """Write a float32 TIFF in raw DN units, undoing the per-band tile rescale.

    The training tiles are produced by clip(DN, 0, c) / c * 65535 (see SOTER's
    rescale_bands_to_uint16). read_tiff then divides by 65535, so the model works
    in [0, 1] = DN / c. This inverts that last step: DN = value * c, per band,
    using the same per-band `ceilings` (pct_ceilings) the tiles were built with.

    `arr` is (H, W, C) in [0, 1]; len(ceilings) must equal C. Output is float32
    (ceilings are floats) and preserves band count. DN is clipped at 0 only — the
    upper bound is the band ceiling, but values can legitimately reach it.
    Geospatial metadata is NOT carried.
    """
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] != len(ceilings):
        msg = (
            f"ceilings length {len(ceilings)} != band count {arr.shape[-1]}; "
            "pick the right --scale-key or check scale.json"
        )
        raise ValueError(msg)
    out = (arr * np.asarray(ceilings, dtype=np.float32)).astype(np.float32)
    if out.shape[-1] == 1:
        out = out[..., 0]
    tifffile.imwrite(str(path), out)
