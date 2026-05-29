"""Standalone multispectral inference for the neosr satellite fork.

Runs a trained generator on N-band uint16 TIFFs and writes N-band uint16 TIFFs,
preserving channel count and 16-bit depth. This deliberately bypasses neosr's
built-in test.py / validation save path, which routes through tensor2img + cv2
and therefore downcasts to 8-bit and mangles >3-band order. Geospatial metadata
(CRS/transform) is NOT preserved — only pixel values, bands, and bit depth.

The model and scale are taken from the SAME training TOML (-opt), so the
generator is built identically to training (esrgan, num_in_ch/out_ch=4, scale=2).

Usage:
    python scripts/inference/infer_multispectral.py \
        -opt options/train_esrgan_satellite_otf.toml \
        --model experiments/train_esrgan_satellite_otf/models/net_g_200000.pth \
        --input  D:/datasets/SOTER/ROI_SR/HD15_16bits_OTF/val/lq \
        --output D:/datasets/SOTER/ROI_SR/HD15_16bits_OTF/val/sr

Input must already be in the model's uint16 tile encoding (read_tiff divides by
65535). To run on raw native imagery, pre-scale it with the per-band ceilings in
scale.json first (same step used to build the training tiles).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="infer_multispectral",
        description="N-band uint16 -> uint16 super-resolution inference.",
    )
    parser.add_argument(
        "-opt", required=True, help="Training TOML (defines arch + scale)."
    )
    parser.add_argument(
        "--model", required=True, help="Path to the trained generator .pth."
    )
    parser.add_argument(
        "--input", required=True, help="Input TIFF file or a directory of TIFFs."
    )
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--param-key",
        default="auto",
        choices=["auto", "params", "params_ema"],
        help="Checkpoint key to load. 'auto' prefers params_ema if present.",
    )
    parser.add_argument(
        "--suffix", default="", help="Optional suffix appended to output filenames."
    )
    args = parser.parse_args()

    # neosr's archs call parse_options() at import time with a strict parse_args(),
    # so hand it only the -opt flag it understands before importing anything.
    sys.argv = [sys.argv[0], "-opt", args.opt]

    import numpy as np
    import torch

    from neosr.archs import build_network
    from neosr.utils.multispectral_io import read_tiff, write_tiff
    from neosr.utils.options import parse_options

    opt, _ = parse_options(str(Path(__file__).resolve().parents[2]), is_train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the generator exactly as configured for training.
    net = build_network(opt["network_g"])
    net = net.to(device).eval()

    # Load weights, mirroring neosr's checkpoint key handling.
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    if args.param_key == "auto":
        for key in ("params_ema", "params-ema", "params"):
            if isinstance(ckpt, dict) and key in ckpt:
                ckpt = ckpt[key]
                break
    elif args.param_key in ckpt:
        ckpt = ckpt[args.param_key]
    ckpt = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)

    # Resolve the input list.
    in_path = Path(args.input)
    if in_path.is_dir():
        files = sorted(
            p for p in in_path.iterdir() if p.suffix.lower() in (".tif", ".tiff")
        )
    else:
        files = [in_path]
    if not files:
        print(f"No .tif/.tiff files found at {in_path}")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(files, 1):
        img = read_tiff(src)  # (H, W, C) float32 in [0, 1]
        tensor = (
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device).float()
        )
        with torch.inference_mode():
            out = net(tensor)
        out_hwc = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        out_name = f"{src.stem}{args.suffix}.tif"
        write_tiff(out_dir / out_name, out_hwc.astype(np.float32))
        print(f"[{i}/{len(files)}] {src.name} -> {out_name}  {out_hwc.shape}")

    print(f"Done. {len(files)} image(s) written to {out_dir}")


if __name__ == "__main__":
    main()
