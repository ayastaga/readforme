"""Image preparation for North Micro Vision.

North Micro Vision processes images at native resolution and its token count
scales with pixel area. The checkpoint's `preprocessor_config.json` advertises a
`longest_edge` far above the 8K-token multimodal context the model was validated
on (see the model card: "validated operating range for multimodal prompts is up
to 8K tokens"). A phone photo of a letter (12 MP+) will blow past that unless we
cap it ourselves. `prepare()` caps pixel area, fixes EXIF orientation and applies a
light auto-contrast that helps photographed (as opposed to scanned) paper.

`MAX_PIXELS` defaults to the A4-at-200dpi ceiling reported by community MLX users
(1654x2339 ~= 3.87 MP). Lower it to trade accuracy for prefill latency.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps

MAX_PIXELS = 1654 * 2339  # ~3.87 MP, A4 @ 200 dpi
EDGE_MULTIPLE = 32


@dataclass
class Prepared:
    image: Image.Image
    original_size: tuple[int, int]
    scale: float


def _snap(v: int) -> int:
    return max(EDGE_MULTIPLE, (v // EDGE_MULTIPLE) * EDGE_MULTIPLE)


def prepare(data: bytes | Image.Image, max_pixels: int = MAX_PIXELS, autocontrast: bool = True) -> Prepared:
    img = data if isinstance(data, Image.Image) else Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    w, h = img.size
    scale = 1.0
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
    nw, nh = _snap(int(w * scale)), _snap(int(h * scale))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
    if autocontrast:
        img = ImageOps.autocontrast(img, cutoff=1)
    return Prepared(image=img, original_size=(w, h), scale=nw / w)


def estimate_tokens(img: Image.Image, patch: int = 16, merge: int = 2) -> int:
    """Rough visual-token estimate: (w/patch/merge) * (h/patch/merge). Used for logging
    and for the eval's pixel-budget sweep; not the exact processor count."""
    w, h = img.size
    return (w // (patch * merge)) * (h // (patch * merge))
