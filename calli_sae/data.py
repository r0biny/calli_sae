import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor


UNKNOWN_CHARACTER = "未知字"
UNKNOWN_STYLE = "未知书体"
UNKNOWN_SOURCE = "未知作者"


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Read UTF-8 CSV metadata rows while tolerating a BOM."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_rel_path(rel_path: str) -> str:
    """Normalize local CSV paths into slash-separated relative paths."""
    return rel_path.replace("\\", "/").strip()


def resolve_target_path(rel_path: str) -> str:
    """Map legacy char/ paths to the processed image folder used by recent experiments."""
    rel_path = normalize_rel_path(rel_path)
    if rel_path.startswith("char_process/"):
        return rel_path
    if rel_path.startswith("char/"):
        return rel_path.replace("char/", "char_process/", 1)
    return rel_path


def path_tags(row: Dict[str, str], image_column: str) -> List[str]:
    rel_path = row.get(image_column, "") or row.get("path", "")
    if not rel_path:
        return []

    basename = os.path.splitext(os.path.basename(normalize_rel_path(rel_path)))[0]
    return basename.split()


def char_from_tag(tag: str) -> str:
    tag = tag.strip()
    if len(tag) > 1 and tag.endswith("字"):
        return tag[:-1]
    return ""


def split_style_text(style_text: str) -> Tuple[str, str]:
    """Split style text into script family and author/source components."""
    style_text = " ".join(style_text.split())
    if not style_text:
        return "", ""

    parts = style_text.split(" ", 1)
    if len(parts) == 1:
        token = parts[0]
        return (token, "") if token.endswith("书") else ("", token)

    first, second = parts
    if first.endswith("书"):
        return first, second
    return "", style_text


@dataclass(frozen=True)
class SampleMetadata:
    character_text: str
    character: str
    style: str
    source: str
    raw: str


def parse_sample_metadata(row: Dict[str, str], image_column: str = "path") -> SampleMetadata:
    """Extract useful debug metadata, preferring explicit CSV columns when present."""
    char = row.get("char", "").strip()
    style_text = row.get("new_text", "").strip()
    style, source = split_style_text(style_text)

    filename_tags = path_tags(row, image_column)
    raw = " ".join(filename_tags)

    if not char and filename_tags:
        char = char_from_tag(filename_tags[0])
    if not style and len(filename_tags) >= 2:
        style = filename_tags[1]
    if not source:
        source = row.get("author", "").strip()
    if not source and len(filename_tags) >= 3:
        source = " ".join(filename_tags[2:])

    char = char or UNKNOWN_CHARACTER
    style = style or UNKNOWN_STYLE
    source = source or UNKNOWN_SOURCE

    character = f"{char}字" if len(char) == 1 else char
    return SampleMetadata(character_text=char, character=character, style=style, source=source, raw=raw)


class CalligraphyCsvDataset(Dataset):
    """CSV-backed single-character calligraphy image dataset."""

    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        dataset_root: Path,
        image_size: int,
        *,
        image_column: str = "path",
        center_crop: bool = True,
        normalize: bool = True,
    ):
        self.rows = list(rows)
        self.dataset_root = Path(dataset_root)
        self.image_column = image_column

        transforms = [
            Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        ]
        if center_crop:
            transforms.append(CenterCrop(image_size))
        transforms.append(ToTensor())
        if normalize:
            transforms.append(Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        self.transform = Compose(transforms)

    def __len__(self) -> int:
        return len(self.rows)

    def resolve_image_path(self, row: Dict[str, str]) -> Path:
        rel_path = row.get(self.image_column, "") or row.get("path", "")
        rel_path = resolve_target_path(rel_path)
        path = Path(rel_path)

        if path.is_absolute():
            return path

        candidates = [
            self.dataset_root / "dataset" / rel_path,
            self.dataset_root / rel_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        image_path = self.resolve_image_path(row)
        image = Image.open(image_path).convert("RGB")
        metadata = parse_sample_metadata(row, self.image_column)

        return {
            "image": self.transform(image),
            "path": normalize_rel_path(row.get(self.image_column, row.get("path", ""))),
            "row_index": idx,
            "char": metadata.character_text,
            "style": metadata.style,
            "source": metadata.source,
            "raw": metadata.raw,
        }


def default_train_csv(dataset_root: Path) -> Path:
    return dataset_root / "dataset" / "train_260402.csv"


def default_val_csv(dataset_root: Path) -> Path:
    validation_csv = dataset_root / "dataset" / "validation.csv"
    legacy_typo_csv = dataset_root / "dataset" / "validaiton.csv"
    return validation_csv if validation_csv.exists() else legacy_typo_csv


def assert_nonempty_rows(rows: Sequence[Dict[str, str]], csv_path: Path) -> None:
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")


def path_like_row_key(path: str) -> str:
    """Return a stable filesystem-safe-ish key for debug files if needed."""
    normalized = normalize_rel_path(path)
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", normalized).strip("_")

