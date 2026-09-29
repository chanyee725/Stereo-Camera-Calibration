"""ChArUco board creation and corner detection."""

import cv2
import numpy as np


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
