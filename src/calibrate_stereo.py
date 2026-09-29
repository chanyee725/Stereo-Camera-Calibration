"""Stereo calibration using the intrinsics from calibrate_mono.py.

Reads <session>/calib/left.yaml and right.yaml, keeps them fixed, and estimates
the relative pose between the cameras from the pairs listed in <session>/pairs.csv.

Outputs in <session>/calib/:
    stereo.yaml          extrinsics and rectification
    rectified_check.png  rectified pair with horizontal lines
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from charuco import (
    DEFAULT_CONFIG,
    build_board,
    detect,
    latest_session,
    load_config,
    read_pairs,
    to_points,
)


def read_intrinsics(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run calibrate_mono.py first")
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    size = (
        int(fs.getNode("image_width").real()),
        int(fs.getNode("image_height").real()),
    )
    fs.release()
    return K, D, size


def rectified_y_error(pts_l, pts_r, K1, D1, R1, P1, K2, D2, R2, P2) -> np.ndarray:
    """Vertical offset in pixels between matching corners after rectification."""
    rect_l = cv2.undistortPoints(pts_l.reshape(-1, 1, 2), K1, D1, R=R1, P=P1)
    rect_r = cv2.undistortPoints(pts_r.reshape(-1, 1, 2), K2, D2, R=R2, P=P2)
    return np.abs(rect_l[:, 0, 1] - rect_r[:, 0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--session", type=Path, help="Capture directory (default: latest in outputs/)"
    )
    parser.add_argument(
        "--pair",
        help="Left image name of the pair for rectified_check.png (default: first)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)["calibration"]
    session = args.session or latest_session()
    calib_dir = session / "calib"
    min_corners = cfg.get("min_corners", 6)

    K1, D1, size_l = read_intrinsics(calib_dir / "left.yaml")
    K2, D2, size_r = read_intrinsics(calib_dir / "right.yaml")
    if size_l != size_r:
        raise ValueError(f"Image size mismatch: left {size_l}, right {size_r}")
    image_size = size_l

    board = build_board(cfg)
    detector = cv2.aruco.CharucoDetector(board)

    names, pairs, obj_points, img_l, img_r = [], [], [], [], []
    for path_l, path_r in read_pairs(session):
        name = f"left/{path_l.name} + right/{path_r.name}"
        det_l = detect(detector, cv2.imread(str(path_l)))
        det_r = detect(detector, cv2.imread(str(path_r)))
        # Only corners seen by both cameras in the same pair are usable
        common = sorted(set(det_l) & set(det_r))
        if len(common) < min_corners:
            print(f"{name}: skipped ({len(common)} common corners)")
            continue
        obj, pts_l = to_points(board, det_l, common)
        _, pts_r = to_points(board, det_r, common)
        names.append(name)
        pairs.append((path_l, path_r))
        obj_points.append(obj)
        img_l.append(pts_l)
        img_r.append(pts_r)
    if not names:
        raise RuntimeError(f"No usable pairs in {session}")

    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_l,
        img_r,
        K1,
        D1,
        K2,
        D2,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, alpha=0
    )

    baseline = float(np.linalg.norm(T))
    angle = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
    y_errors = [
        rectified_y_error(pl, pr, K1, D1, R1, P1, K2, D2, R2, P2)
        for pl, pr in zip(img_l, img_r)
    ]
    all_y = np.concatenate(y_errors)

    print(f"\nStereo RMS: {rms:.4f} px ({len(names)} pairs)")
    print(f"Baseline: {baseline:.2f} mm (expected {cfg['expected_baseline_mm']} mm)")
    print(f"T (mm): {np.round(T.ravel(), 2)}")
    print(f"Rotation between cameras: {angle:.2f} deg")
    print(f"Rectified y error: mean {all_y.mean():.3f} px, max {all_y.max():.3f} px")
    median = np.median([e.mean() for e in y_errors])
    for name, err in zip(names, y_errors):
        flag = "  <-- check" if err.mean() > 2 * median else ""
        print(f"  {name}: mean {err.mean():.3f} px, max {err.max():.3f} px{flag}")

    fs = cv2.FileStorage(str(calib_dir / "stereo.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("rms", rms)
    fs.write("baseline_mm", baseline)
    fs.write("rectified_y_error_mean", float(all_y.mean()))
    for key, value in [
        ("K1", K1), ("D1", D1), ("K2", K2), ("D2", D2),
        ("R", R), ("T", T), ("E", E), ("F", F),
        ("R1", R1), ("R2", R2), ("P1", P1), ("P2", P2), ("Q", Q),
    ]:  # fmt: skip
        fs.write(key, value)
    fs.release()

    if args.pair:
        matches = [i for i, (pl, _) in enumerate(pairs) if pl.name == args.pair]
        if not matches:
            raise ValueError(f"{args.pair} is not a usable left image of any pair")
        path_l, path_r = pairs[matches[0]]
    else:
        path_l, path_r = pairs[0]
    pair = f"left/{path_l.name} + right/{path_r.name}"
    map_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)
    left = cv2.imread(str(path_l))
    right = cv2.imread(str(path_r))
    check = np.hstack(
        (
            cv2.remap(left, *map_l, cv2.INTER_LINEAR),
            cv2.remap(right, *map_r, cv2.INTER_LINEAR),
        )
    )
    for y in range(0, check.shape[0], 40):
        cv2.line(check, (0, y), (check.shape[1], y), (0, 255, 0), 1)
    cv2.putText(check, pair, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(calib_dir / "rectified_check.png"), check)
    print(f"Saved results to {calib_dir} (rectified_check.png uses {pair})")


if __name__ == "__main__":
    main()
