"""
Grad-CAM for the EfficientNet-B0 WBC classifier.

A self-contained, dependency-free implementation (we already use torch + PIL).
Hooks the activations and gradients of the last conv stage and produces a
per-image attention heatmap, returned as a base64-encoded PNG overlay.

The target layer defaults to ``model.conv_head`` (the 1x1 expansion that
follows the final inverted-residual block in timm's ``efficientnet_b0``).
This is a stable, well-known choice for EfficientNet Grad-CAM.

References
----------
Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# Simple jet-like colormap (256x3, RGB float in [0,1]) — avoids matplotlib dep.
def _build_jet_lut() -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=1)


_JET = _build_jet_lut()


def _apply_jet(cam01: np.ndarray) -> np.ndarray:
    """Map a [H,W] float array in [0,1] to [H,W,3] uint8 RGB via JET."""
    idx = np.clip((cam01 * 255.0).astype(np.int32), 0, 255)
    return (_JET[idx] * 255.0).astype(np.uint8)


class GradCAM:
    """
    Grad-CAM extractor for a CNN classifier.

    Usage::

        cam = GradCAM(model, target_layer=model.conv_head)
        heatmap = cam.compute(input_tensor, class_idx=predicted_idx)  # [H, W] in [0, 1]
        overlay_b64 = cam.render_overlay(pil_crop, heatmap)           # data:image/png;base64,...
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer if target_layer is not None else self._auto_target(model)
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    @staticmethod
    def _auto_target(model: nn.Module) -> nn.Module:
        # timm efficientnet exposes `.conv_head`; otherwise fall back to the
        # last Conv2d layer in the module tree.
        if hasattr(model, "conv_head"):
            return model.conv_head  # type: ignore[return-value]
        last_conv: Optional[nn.Module] = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No suitable target layer found for Grad-CAM.")
        return last_conv

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def _save_activations(self, _module, _inp, out) -> None:  # type: ignore[no-untyped-def]
        self._activations = out.detach()

    def _save_gradients(self, _module, _grad_in, grad_out) -> None:  # type: ignore[no-untyped-def]
        self._gradients = grad_out[0].detach()

    @torch.enable_grad()
    def compute(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Compute the Grad-CAM heatmap (float32 in [0,1], H_in x W_in)."""
        was_training = self.model.training
        self.model.eval()
        # Important: re-enable grads for params during the forward pass.
        self.model.zero_grad(set_to_none=True)
        x = input_tensor.detach().clone().requires_grad_(False)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")

        # weights: global-average-pool of gradients across spatial dims -> [C]
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))  # [1, 1, H, W]

        # Upsample to input resolution.
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam[0, 0].cpu().numpy().astype(np.float32)

        # Normalise to [0, 1].
        cam_np -= cam_np.min()
        max_val = cam_np.max()
        if max_val > 1e-8:
            cam_np /= max_val
        else:
            cam_np[:] = 0.0

        if was_training:
            self.model.train()
        return cam_np

    @staticmethod
    def render_overlay(
        crop: Union[np.ndarray, Image.Image],
        heatmap01: np.ndarray,
        alpha: float = 0.45,
    ) -> str:
        """Blend `crop` (RGB) with a JET-colored `heatmap01` and return a data URL."""
        if isinstance(crop, np.ndarray):
            base = Image.fromarray(crop.astype(np.uint8)).convert("RGB")
        else:
            base = crop.convert("RGB")

        if heatmap01.shape[:2] != (base.size[1], base.size[0]):
            heatmap_img = Image.fromarray((heatmap01 * 255).astype(np.uint8)).resize(
                base.size, resample=Image.BILINEAR
            )
            heatmap01 = np.asarray(heatmap_img, dtype=np.float32) / 255.0

        rgb = _apply_jet(heatmap01)
        heat = Image.fromarray(rgb, mode="RGB")
        blended = Image.blend(base, heat, alpha=float(alpha))

        buf = io.BytesIO()
        blended.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
