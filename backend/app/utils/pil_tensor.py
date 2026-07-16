"""PIL → torch tensor helpers (avoid torchvision ToTensor / torch.from_numpy ABI issues)."""

from __future__ import annotations

import torch
from PIL import Image


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert RGB PIL image to float CHW tensor scaled to [0, 1]."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    buffer = rgb.tobytes()
    tensor = torch.frombuffer(bytearray(buffer), dtype=torch.uint8).reshape(height, width, 3)
    return tensor.permute(2, 0, 1).contiguous().float().mul_(1.0 / 255.0)
