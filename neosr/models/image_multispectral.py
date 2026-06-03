"""Multispectral SR model with N-band-faithful validation.

The stock `image` model's validation routes the SR/GT tensors through
`tensor2img`, which downcasts to 8-bit and keeps at most 3 channels. For N-band
satellite tensors that silently drops NIR and crushes 16-bit precision, so any
PSNR/SSIM computed on it is meaningless and the saved previews are lossy.

This subclass overrides only `nondist_validation` to:
  * compute metrics directly on the float (N, C, H, W) tensors — MS-SSIM (via the
    existing `mssim_loss`, which is channel-agnostic) plus mean and per-band PSNR;
  * save N-band uint16 TIFF previews via `write_tiff` instead of 8-bit PNGs.

The training path is untouched, so loss/optimizer/GAN behaviour and throughput are
identical to `image`. Select it with `model_type = "image_multispectral"`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from neosr.losses.ssim_loss import mssim_loss
from neosr.models.image import image
from neosr.utils import tc
from neosr.utils.multispectral_io import write_tiff
from neosr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class image_multispectral(image):
    """SISR model whose validation is N-band- and 16-bit-faithful."""

    # Band labels for per-band PSNR logging; falls back to the index past these.
    _BAND_NAMES = ("R", "G", "B", "NIR")

    def _get_mssim(self, channels: int) -> mssim_loss:
        # Cache one mssim_loss on the model device. L=1 because the tiles are in
        # [0, 1] (read_tiff divides by 65535). It is a *loss* (1 - MS-SSIM), so the
        # metric is 1 - forward().
        metric = getattr(self, "_mssim_metric", None)
        if metric is None:
            metric = mssim_loss(in_channels=channels, L=1).to(self.device)
            self._mssim_metric = metric
        return metric

    def nondist_validation(
        self, dataloader, current_iter: int, tb_logger, save_img: bool = True
    ) -> None:
        # flag to not apply augmentation during val (mirrors image.nondist_validation)
        self.is_train = False
        dataset_name = dataloader.dataset.opt["name"]
        dataset_type = dataloader.dataset.opt["type"]
        use_pbar = self.opt["val"].get("pbar", True)

        # "single" datasets have no GT, so no reference metrics are possible.
        with_metrics = (
            dataset_type != "single" and self.opt["val"].get("metrics") is not None
        )

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only on the first run
                self.metric_results: dict[str, float] = dict.fromkeys(
                    self.opt["val"]["metrics"].keys(), 0
                )
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = dict.fromkeys(self.metric_results, 0)

        # Per-band PSNR is logged to TensorBoard but not best-tracked (kept out of
        # metric_results so _initialize_best_metric_results doesn't need it).
        band_psnr_sum: dict[str, float] = {}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image", colour="green", ascii=" >=")

        num_img = 0
        for val_data in dataloader:
            img_name = Path(val_data["lq_path"][0]).stem
            self.feed_data(val_data)

            model = (
                self.net_g_ema
                if (hasattr(self, "ema") and self.ema > 0)
                else self.net_g
            )
            sf_mode = self.sf_optim_g and self.is_train
            # set eval mode
            model.eval()
            if sf_mode:
                self.optimizer_g.eval()
            # inference
            tile_opt = self.opt["val"].get("tile", -1)
            with torch.inference_mode():
                self.output = self.tile_val() if tile_opt != -1 else model(self.lq)
            # set train mode
            model.train()
            if sf_mode:
                self.optimizer_g.train()

            num_img += 1

            # ---- metrics on the float tensors (clamped to [0, 1] like the output) ----
            if with_metrics and hasattr(self, "gt"):
                with torch.inference_mode():
                    sr = self.output.detach().clamp(0.0, 1.0).float()
                    gt = self.gt.detach().clamp(0.0, 1.0).float()
                    if "mssim" in self.metric_results:
                        ms = self._get_mssim(sr.shape[1])
                        self.metric_results["mssim"] += float(1.0 - ms(sr, gt))
                    mse_per_band = ((sr - gt) ** 2).mean(dim=(0, 2, 3))  # (C,)
                    psnr_per_band = 10.0 * torch.log10(
                        1.0 / mse_per_band.clamp_min(1e-12)
                    )
                    if "psnr" in self.metric_results:
                        self.metric_results["psnr"] += float(psnr_per_band.mean())
                    for c in range(sr.shape[1]):
                        band = (
                            self._BAND_NAMES[c]
                            if c < len(self._BAND_NAMES)
                            else str(c)
                        )
                        band_psnr_sum[band] = band_psnr_sum.get(band, 0.0) + float(
                            psnr_per_band[c]
                        )

            # ---- save an N-band uint16 TIFF preview (not an 8-bit PNG) ----
            if save_img:
                out_hwc = (
                    self.output.detach()
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .clamp(0.0, 1.0)
                    .cpu()
                    .numpy()
                )
                v_folder = self.opt["path"]["visualization"]
                if self.opt["is_train"]:
                    save_path = (
                        Path(v_folder) / img_name / f"{img_name}_{current_iter}.tif"
                    )
                else:
                    suffix = self.opt["val"].get("suffix") or self.opt["name"]
                    save_path = Path(v_folder) / dataset_name / f"{img_name}_{suffix}.tif"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                write_tiff(save_path, out_hwc.astype(np.float32))

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if hasattr(self, "gt"):
                del self.gt
            torch.cuda.empty_cache()

            if use_pbar:
                pbar.update(1)  # type: ignore[reportPossiblyUnboundVariable]
                pbar.set_description(f"{tc.light_green}Inferring on {img_name}{tc.end}")  # type: ignore[reportPossiblyUnboundVariable]

        if use_pbar:
            pbar.close()  # type: ignore[reportPossiblyUnboundVariable]

        if with_metrics and num_img > 0:
            for metric in self.metric_results:
                # correct mean over the validation set (stock image.py has a
                # precedence bug here: `/ _idx + 1`)
                self.metric_results[metric] = self.metric_results[metric] / num_img
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            if tb_logger:
                for band, total in band_psnr_sum.items():
                    tb_logger.add_scalar(
                        f"metrics/{dataset_name}/psnr_{band}",
                        total / num_img,
                        current_iter,
                    )

        self.is_train = True
