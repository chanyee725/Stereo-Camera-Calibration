"""Reading calibration results written by the calibration scripts."""

from pathlib import Path

import cv2


def read_intrinsics(path: Path):
    """Return (K, D, (width, height)) from a MonoCalibration result."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run `calibrate.py mono` first")
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    size = (
        int(fs.getNode("image_width").real()),
        int(fs.getNode("image_height").real()),
    )
    fs.release()
    return K, D, size
