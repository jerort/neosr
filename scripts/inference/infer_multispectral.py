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

By default the output is uint16 in the same tile encoding as the input. Pass
--scale-json to instead write float32 TIFFs in raw DN units (undoing the per-band
ceiling rescale: DN = value * ceiling), which is what the rater expects:
    ... --scale-json D:/datasets/SOTER/ROI_SR/HD15_16bits_OTF/gt/scale.json \
        --scale-key <hd15_source_filename>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_ceilings(scale_json: str, scale_key: str | None) -> list[float]:
    """Resolve per-band ceilings from scale.json for raw-DN output.

    Accepts the SOTER manifest {"images": {<name>: {"pct_ceilings": [...]}}}, a
    single entry {"pct_ceilings": [...]}, or a bare list [c0, c1, ...]. With the
    manifest form, --scale-key selects the entry; it may be omitted only when
    there is exactly one image.
    """
    import json

    with open(scale_json) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [float(c) for c in data]

    if "images" in data:
        images = data["images"]
        if scale_key is None:
            if len(images) != 1:
                keys = ", ".join(sorted(images))
                msg = f"scale.json has {len(images)} images; pass --scale-key one of: {keys}"
                raise SystemExit(msg)
            entry = next(iter(images.values()))
        elif scale_key in images:
            entry = images[scale_key]
        else:
            keys = ", ".join(sorted(images))
            raise SystemExit(f"--scale-key '{scale_key}' not in scale.json. Available: {keys}")
        return [float(c) for c in entry["pct_ceilings"]]

    if "pct_ceilings" in data:
        return [float(c) for c in data["pct_ceilings"]]

    raise SystemExit(
        "Unrecognised scale.json shape: expected a list, {'pct_ceilings': [...]}, "
        "or {'images': {...}}."
    )


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
    parser.add_argument(
        "--scale-json",
        default=None,
        help="Path to scale.json. Enables raw-DN float32 output (DN = value * ceiling).",
    )
    parser.add_argument(
        "--scale-key",
        default=None,
        help="Which image entry in scale.json['images'] to use. Optional when there "
        "is exactly one entry, or when scale.json is a flat ceilings list.",
    )
    args = parser.parse_args()

    # Make the repo root importable when run as a file from scripts/inference/.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # neosr's archs call parse_options() at import time with a strict parse_args(),
    # so hand it only the -opt flag it understands before importing anything.
    sys.argv = [sys.argv[0], "-opt", args.opt]

    import numpy as np
    import torch

    from neosr.archs import build_network
    from neosr.utils.multispectral_io import read_tiff, write_tiff, write_tiff_dn
    from neosr.utils.options import parse_options

    # Raw-DN mode: resolve the per-band ceilings up front so we fail before inference.
    ceilings = _load_ceilings(args.scale_json, args.scale_key) if args.scale_json else None

    opt, _ = parse_options(str(repo_root), is_train=False)

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
        if ceilings is not None:
            write_tiff_dn(out_dir / out_name, out_hwc.astype(np.float32), ceilings)
        else:
            write_tiff(out_dir / out_name, out_hwc.astype(np.float32))
        print(f"[{i}/{len(files)}] {src.name} -> {out_name}  {out_hwc.shape}")

    units = "raw DN (float32)" if ceilings is not None else "uint16 tile encoding"
    print(f"Done. {len(files)} image(s) written to {out_dir} as {units}.")


if __name__ == "__main__":
    main()
