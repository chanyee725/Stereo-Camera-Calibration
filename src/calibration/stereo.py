"""Stereo extrinsics using the fixed intrinsics from MonoCalibration.

Reads <session>/calib/left.yaml and right.yaml and estimates the relative pose
between the cameras from the pairs listed in <session>/pairs.csv.

Outputs in <session>/calib/:
    stereo.yaml          extrinsics and rectification
    rectified_check.png  rectified pair with horizontal lines
"""

from pathlib import Path

import cv2
import numpy as np

from calibration.base import Calibration
from utils.board import to_points
from utils.calib_io import read_intrinsics
from utils.draw import draw_hlines, draw_label
from utils.metrics import rectified_y_error
from utils.session import read_pairs


class StereoCalibration(Calibration):
    def __init__(self, cfg: dict, session: Path, check_pair: str = None):
        super().__init__(cfg, session)
        self.check_pair = check_pair
        self.K1, self.D1, size_l = read_intrinsics(self.out_dir / "left.yaml")
        self.K2, self.D2, size_r = read_intrinsics(self.out_dir / "right.yaml")
        if size_l != size_r:
            raise ValueError(f"Image size mismatch: left {size_l}, right {size_r}")
        self.image_size = size_l

    def collect(self):
        self.names, self.pairs = [], []
        self.obj_points, self.img_l, self.img_r = [], [], []
        for path_l, path_r in read_pairs(self.session):
            name = f"left/{path_l.name} + right/{path_r.name}"
            det_l = self.detect(cv2.imread(str(path_l)))
            det_r = self.detect(cv2.imread(str(path_r)))
            # Only corners seen by both cameras in the same pair are usable
            common = sorted(set(det_l) & set(det_r))
            if len(common) < self.min_corners:
                print(f"{name}: skipped ({len(common)} common corners)")
                continue
            obj, pts_l = to_points(self.board, det_l, common)
            _, pts_r = to_points(self.board, det_r, common)
            self.names.append(name)
            self.pairs.append((path_l, path_r))
            self.obj_points.append(obj)
            self.img_l.append(pts_l)
            self.img_r.append(pts_r)
        if not self.names:
            raise RuntimeError(f"No usable pairs in {self.session}")

    def calibrate(self):
        self.rms, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.obj_points,
            self.img_l,
            self.img_r,
            self.K1,
            self.D1,
            self.K2,
            self.D2,
            self.image_size,
            flags=cv2.CALIB_FIX_INTRINSIC,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.K1, self.D1, self.K2, self.D2, self.image_size, self.R, self.T,
            alpha=0,
        )  # fmt: skip
        self.baseline = float(np.linalg.norm(self.T))
        self.y_errors = [
            rectified_y_error(
                pl, pr, self.K1, self.D1, self.R1, self.P1,
                self.K2, self.D2, self.R2, self.P2,
            )  # fmt: skip
            for pl, pr in zip(self.img_l, self.img_r)
        ]
        self.all_y = np.concatenate(self.y_errors)

    def report(self):
        angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(self.R)[0])))
        expected = self.cfg["expected_baseline_mm"]
        print(f"\nStereo RMS: {self.rms:.4f} px ({len(self.names)} pairs)")
        print(f"Baseline: {self.baseline:.2f} mm (expected {expected} mm)")
        print(f"T (mm): {np.round(self.T.ravel(), 2)}")
        print(f"Rotation between cameras: {angle:.2f} deg")
        print(
            f"Rectified y error: mean {self.all_y.mean():.3f} px, "
            f"max {self.all_y.max():.3f} px"
        )
        median = np.median([e.mean() for e in self.y_errors])
        for name, err in zip(self.names, self.y_errors):
            flag = self.outlier_flag(err.mean(), median)
            print(f"  {name}: mean {err.mean():.3f} px, max {err.max():.3f} px{flag}")

    def save(self):
        self.write_yaml(
            "stereo.yaml",
            {
                "rms": self.rms,
                "baseline_mm": self.baseline,
                "rectified_y_error_mean": float(self.all_y.mean()),
                "K1": self.K1, "D1": self.D1, "K2": self.K2, "D2": self.D2,
                "R": self.R, "T": self.T, "E": self.E, "F": self.F,
                "R1": self.R1, "R2": self.R2, "P1": self.P1, "P2": self.P2,
                "Q": self.Q,
            },
        )  # fmt: skip
        path_l, path_r = self.select_check_pair()
        pair = f"left/{path_l.name} + right/{path_r.name}"
        cv2.imwrite(
            str(self.out_dir / "rectified_check.png"),
            self.draw_rectified(path_l, path_r, pair),
        )
        print(f"Saved results to {self.out_dir} (rectified_check.png uses {pair})")

    def select_check_pair(self):
        if not self.check_pair:
            return self.pairs[0]
        for path_l, path_r in self.pairs:
            if path_l.name == self.check_pair:
                return path_l, path_r
        raise ValueError(f"{self.check_pair} is not a usable left image of any pair")

    def draw_rectified(self, path_l: Path, path_r: Path, label: str):
        map_l = cv2.initUndistortRectifyMap(
            self.K1, self.D1, self.R1, self.P1, self.image_size, cv2.CV_32FC1
        )
        map_r = cv2.initUndistortRectifyMap(
            self.K2, self.D2, self.R2, self.P2, self.image_size, cv2.CV_32FC1
        )
        check = np.hstack(
            (
                cv2.remap(cv2.imread(str(path_l)), *map_l, cv2.INTER_LINEAR),
                cv2.remap(cv2.imread(str(path_r)), *map_r, cv2.INTER_LINEAR),
            )
        )
        draw_hlines(check)
        draw_label(check, label)
        return check
