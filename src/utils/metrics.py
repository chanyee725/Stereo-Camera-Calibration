"""Quality metrics for calibration results."""

import cv2
import numpy as np

COVERAGE_GRID = (8, 6)


def coverage_ratio(points, image_size) -> float:
    """Fraction of COVERAGE_GRID cells that contain at least one corner."""
    w, h = image_size
    gx, gy = COVERAGE_GRID
    cells = {(int(x * gx / w), int(y * gy / h)) for x, y in points}
    return len(cells) / (gx * gy)


def rectified_y_error(pts_l, pts_r, K1, D1, R1, P1, K2, D2, R2, P2) -> np.ndarray:
    """Vertical offset in pixels between matching corners after rectification."""
    rect_l = cv2.undistortPoints(pts_l.reshape(-1, 1, 2), K1, D1, R=R1, P=P1)
    rect_r = cv2.undistortPoints(pts_r.reshape(-1, 1, 2), K2, D2, R=R2, P=P2)
    return np.abs(rect_l[:, 0, 1] - rect_r[:, 0, 1])
