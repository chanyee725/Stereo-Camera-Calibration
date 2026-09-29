"""Capture session layout: image folders, numbering and stereo pair records."""

import csv
from pathlib import Path

import cv2

from utils.config import ROOT

OUTPUT_ROOT = ROOT / "outputs"
IMAGE_DIRS = ("left", "right")
# Left and right are numbered independently, so stereo pairs are recorded here
PAIRS_FILE = "pairs.csv"


def latest_session() -> Path:
    sessions = []
    if OUTPUT_ROOT.is_dir():
        sessions = sorted(
            p
            for p in OUTPUT_ROOT.iterdir()
            if any((p / d).is_dir() for d in IMAGE_DIRS)
        )
    if not sessions:
        raise FileNotFoundError(
            f"No capture sessions found in {OUTPUT_ROOT}, run acquire_image.py first"
        )
    return sessions[-1]


def list_images(directory: Path) -> list:
    return sorted(directory.glob("*.png"))


def last_index(directory: Path) -> int:
    indices = [int(p.stem) for p in directory.glob("*.png") if p.stem.isdigit()]
    return max(indices, default=0)


def save_image(directory: Path, index: int, image) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{index:02d}.png"
    cv2.imwrite(str(directory / name), image)
    return name


def append_pair(session: Path, left_name: str, right_name: str):
    path = session / PAIRS_FILE
    is_new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        if is_new:
            writer.writerow(["left", "right"])
        writer.writerow([left_name, right_name])


def read_pairs(session: Path) -> list:
    """Return [(left_path, right_path), ...] recorded by acquire_image.py."""
    path = session / PAIRS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, capture pairs with the 's' key")
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        (session / "left" / r["left"], session / "right" / r["right"]) for r in rows
    ]
