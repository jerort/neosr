"""Multispectral TIFF IO for neosr satellite-imagery fork.

Bypasses cv2.imdecode (which forces 3-ch BGR uint8) and returns float32
arrays normalized to [0, 1] via /65535.0, preserving full 16-bit depth.

Band selection is treated as a preprocessing step outside training: this
loader returns whatever bands are present in the file. The model's
num_in_ch / num_out_ch in the training config must match.
"""

from __future__ import annotations

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
