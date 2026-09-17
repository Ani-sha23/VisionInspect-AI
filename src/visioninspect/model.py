"""U-Net definition and checkpoint loading.

torch is an optional dependency: the classical baseline path must work on a
machine with no deep-learning stack installed, so torch is imported lazily and
its absence raises a typed ModelError instead of an ImportError at import time.
"""

from __future__ import annotations

import os
from typing import Any

from .exceptions import ModelError


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except Exception as exc:  # pragma: no cover - exercised only without torch
        raise ModelError(
            "PyTorch is not installed. Install it (pip install torch) to use the U-Net "
            "backend, or run the pipeline with --backend baseline."
        ) from exc


def build_unet(in_channels: int = 3, num_classes: int = 4, base_filters: int = 32) -> Any:
    """Return a compact U-Net with a 4-channel sigmoid output (multi-label segmentation)."""
    torch, nn = _require_torch()

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            f = base_filters
            self.enc1 = block(in_channels, f)
            self.enc2 = block(f, f * 2)
            self.enc3 = block(f * 2, f * 4)
            self.enc4 = block(f * 4, f * 8)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = block(f * 8, f * 16)
            self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
            self.dec4 = block(f * 16, f * 8)
            self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
            self.dec3 = block(f * 8, f * 4)
            self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
            self.dec2 = block(f * 4, f * 2)
            self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
            self.dec1 = block(f * 2, f)
            self.head = nn.Conv2d(f, num_classes, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)  # logits; apply sigmoid outside

    return UNet()


def resolve_device(preference: str = "auto"):
    torch, _ = _require_torch()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ModelError("device: cuda requested but CUDA is not available on this machine.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(weights_path: str, cfg_model: dict, device_preference: str = "auto"):
    """Load a trained U-Net. Raises ModelError if the file is absent or incompatible."""
    if not os.path.isfile(weights_path):
        raise ModelError(f"Model weights not found at '{weights_path}'.")
    torch, _ = _require_torch()
    device = resolve_device(device_preference)
    model = build_unet(
        int(cfg_model.get("in_channels", 3)),
        int(cfg_model.get("num_classes", 4)),
        int(cfg_model.get("base_filters", 32)),
    )
    try:
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state)
    except Exception as exc:
        raise ModelError(f"Checkpoint at '{weights_path}' does not match the configured U-Net: {exc}") from exc
    model.to(device).eval()
    return model, device
