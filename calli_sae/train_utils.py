import random
from pathlib import Path
from typing import Optional

import torch
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """Convert images from [-1, 1] to [0, 1]."""
    return (images / 2.0 + 0.5).clamp(0.0, 1.0)


def save_reconstruction_grid(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    path: Path,
    *,
    max_samples: Optional[int] = None,
) -> None:
    """Save a two-column original/reconstruction grid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if max_samples is not None:
        originals = originals[:max_samples]
        reconstructions = reconstructions[:max_samples]

    rows = []
    for original, recon in zip(originals, reconstructions):
        rows.extend([original.detach().cpu(), recon.detach().cpu()])

    if not rows:
        raise ValueError("Cannot save an empty reconstruction grid.")

    grid = torch.stack(rows)
    save_image(denormalize_images(grid), path, nrow=2)

