"""Shared helpers for the ChArUco calibration scripts."""

import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
OUTPUT_ROOT = ROOT / "outputs"
IMAGE_DIRS = ("left", "right")
# Left and right are numbered independently, so stereo pairs are recorded here
PAIRS_FILE = "pairs.csv"


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def build_board(cfg: dict):
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, cfg["aruco_dict"])
    )
    return cv2.aruco.CharucoBoard(
        (cfg["squares_x"], cfg["squares_y"]),
        cfg["square_size_mm"],
        cfg["marker_size_mm"],
        dictionary,
    )


def detect(detector, image) -> dict:
    """Return {corner_id: (x, y)} for the ChArUco corners found in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None:
        return {}
    return {int(i): c for i, c in zip(ids.flatten(), corners.reshape(-1, 2))}


def to_points(board, detection: dict, ids=None):
    ids = sorted(detection) if ids is None else ids
    obj = board.getChessboardCorners()[ids].astype(np.float32)
    img = np.array([detection[i] for i in ids], dtype=np.float32)
    return obj, img
