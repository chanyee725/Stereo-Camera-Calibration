"""Single camera intrinsics.

Outputs in <session>/calib/:
    <side>.yaml           intrinsics and per-view reprojection errors
    <side>_coverage.png   detected corners of all views drawn on one frame
    <side>_undistort.png  original (left) vs undistorted (right) sample
"""

from pathlib import Path

import cv2
import numpy as np

from calibration.base import Calibration
from utils.board import to_points
from utils.constants import COVERAGE_GRID, GREEN
from utils.draw import draw_grid, draw_label
from utils.metrics import coverage_ratio
from utils.session import list_images


class MonoCalibration(Calibration):
    def __init__(self, cfg: dict, session: Path, side: str, image_dirs=None):
        super().__init__(cfg, session)
        self.side = side
        self.image_dirs = image_dirs or [session / side]

    def collect(self):
        self.names, self.obj_points, self.img_points = [], [], []
        self.sample = None
        for path in (p for d in self.image_dirs for p in list_images(d)):
            image = cv2.imread(str(path))
            self.image_size = (image.shape[1], image.shape[0])
            self.sample = image if self.sample is None else self.sample
            detection = self.detect(image)
            if len(detection) < self.min_corners:
                print(
                    f"{path.parent.name}/{path.name}: skipped ({len(detection)} corners)"
                )
                continue
            obj, img = to_points(self.board, detection)
            self.names.append(f"{path.parent.name}/{path.name}")
            self.obj_points.append(obj)
            self.img_points.append(img)
        if not self.names:
            raise RuntimeError(
                f"No usable views in {[str(d) for d in self.image_dirs]}"
            )

    def calibrate(self):
        flags = cv2.CALIB_FIX_K3 if self.cfg.get("fix_k3", False) else 0
        self.rms, self.K, self.D, _, _, self.std, _, view_errors = (
            cv2.calibrateCameraExtended(
                self.obj_points,
                self.img_points,
                self.image_size,
                None,
                None,
                flags=flags,
            )
        )
        self.view_errors = view_errors.ravel()
        self.all_points = np.vstack(self.img_points)
        self.coverage = coverage_ratio(self.all_points, self.image_size)

    def report(self):
        K, std = self.K, self.std
        print(f"\n[{self.side}] RMS: {self.rms:.4f} px ({len(self.names)} views)")
        print(f"fx={K[0, 0]:.1f} ± {std[0, 0]:.1f}  fy={K[1, 1]:.1f} ± {std[1, 0]:.1f}")
        print(f"cx={K[0, 2]:.1f} ± {std[2, 0]:.1f}  cy={K[1, 2]:.1f} ± {std[3, 0]:.1f}")
        print(f"dist (k1 k2 p1 p2 k3): {np.round(self.D.ravel(), 4)}")
        gx, gy = COVERAGE_GRID
        print(f"coverage: {self.coverage * 100:.0f}% of {gx}x{gy} grid")
        median = np.median(self.view_errors)
        for name, err in zip(self.names, self.view_errors):
            print(f"  {name}: {err:.4f} px{self.outlier_flag(err, median)}")

    def save(self):
        self.write_yaml(
            f"{self.side}.yaml",
            {
                "rms": self.rms,
                "coverage": self.coverage,
                "K": self.K,
                "D": self.D,
                "std_intrinsics": self.std[:9],
                "per_view_errors": self.view_errors,
                "view_names": self.names,
            },
        )
        cv2.imwrite(
            str(self.out_dir / f"{self.side}_coverage.png"), self.draw_coverage()
        )
        cv2.imwrite(
            str(self.out_dir / f"{self.side}_undistort.png"),
            np.hstack((self.sample, cv2.undistort(self.sample, self.K, self.D))),
        )
        print(f"Saved results to {self.out_dir}")

    def draw_coverage(self):
        canvas = (self.sample * 0.5).astype(np.uint8)
        draw_grid(canvas, COVERAGE_GRID)
        for x, y in self.all_points:
            cv2.circle(canvas, (int(x), int(y)), 2, GREEN, -1)
        draw_label(canvas, f"coverage: {self.coverage * 100:.0f}%")
        return canvas
