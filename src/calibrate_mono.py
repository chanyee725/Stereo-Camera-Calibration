"""Calibrate a single camera from ChArUco images and check the result.

Outputs in <session>/calib/:
    <side>.yaml           intrinsics and per-view reprojection errors
    <side>_coverage.png   detected corners of all views drawn on one frame
    <side>_undistort.png  original (left) vs undistorted (right) sample
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from utils.board import build_board, detect, to_points
from utils.config import DEFAULT_CONFIG, load_config
from utils.metrics import COVERAGE_GRID, coverage_ratio
from utils.session import latest_session, list_images


def draw_coverage(image, points, image_size, ratio):
    canvas = (image * 0.5).astype(np.uint8)
    w, h = image_size
    gx, gy = COVERAGE_GRID
    for i in range(1, gx):
        cv2.line(canvas, (i * w // gx, 0), (i * w // gx, h), (128, 128, 128), 1)
    for i in range(1, gy):
        cv2.line(canvas, (0, i * h // gy), (w, i * h // gy), (128, 128, 128), 1)
    for x, y in points:
        cv2.circle(canvas, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.putText(
        canvas,
        f"coverage: {ratio * 100:.0f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", required=True, choices=["left", "right"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--session", type=Path, help="Capture directory (default: latest in outputs/)"
    )
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        help="Image directories (default: <session>/<side>)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)["calibration"]
    session = args.session or latest_session()
    image_dirs = args.images or [session / args.side]
    images = [path for d in image_dirs for path in list_images(d)]
    min_corners = cfg.get("min_corners", 6)

    board = build_board(cfg)
    detector = cv2.aruco.CharucoDetector(board)

    names, obj_points, img_points = [], [], []
    image_size, sample = None, None
    for path in images:
        image = cv2.imread(str(path))
        image_size = (image.shape[1], image.shape[0])
        sample = image if sample is None else sample
        detection = detect(detector, image)
        if len(detection) < min_corners:
            print(f"{path.parent.name}/{path.name}: skipped ({len(detection)} corners)")
            continue
        obj, img = to_points(board, detection)
        names.append(f"{path.parent.name}/{path.name}")
        obj_points.append(obj)
        img_points.append(img)
    if not names:
        raise RuntimeError(f"No usable views in {[str(d) for d in image_dirs]}")

    flags = cv2.CALIB_FIX_K3 if cfg.get("fix_k3", False) else 0
    rms, K, D, _, _, std, _, view_errors = cv2.calibrateCameraExtended(
        obj_points, img_points, image_size, None, None, flags=flags
    )
    view_errors = view_errors.ravel()
    all_points = np.vstack(img_points)
    ratio = coverage_ratio(all_points, image_size)

    print(f"\n[{args.side}] RMS: {rms:.4f} px ({len(names)} views)")
    print(f"fx={K[0, 0]:.1f} ± {std[0, 0]:.1f}  fy={K[1, 1]:.1f} ± {std[1, 0]:.1f}")
    print(f"cx={K[0, 2]:.1f} ± {std[2, 0]:.1f}  cy={K[1, 2]:.1f} ± {std[3, 0]:.1f}")
    print(f"dist (k1 k2 p1 p2 k3): {np.round(D.ravel(), 4)}")
    print(f"coverage: {ratio * 100:.0f}% of {COVERAGE_GRID[0]}x{COVERAGE_GRID[1]} grid")
    median = np.median(view_errors)
    for name, err in zip(names, view_errors):
        flag = "  <-- check" if err > 2 * median else ""
        print(f"  {name}: {err:.4f} px{flag}")

    out_dir = session / "calib"
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(out_dir / f"{args.side}.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("rms", rms)
    fs.write("coverage", ratio)
    fs.write("K", K)
    fs.write("D", D)
    fs.write("std_intrinsics", std[:9])
    fs.write("per_view_errors", view_errors)
    fs.startWriteStruct("view_names", cv2.FileNode_SEQ)
    for name in names:
        fs.write("", name)
    fs.endWriteStruct()
    fs.release()

    cv2.imwrite(
        str(out_dir / f"{args.side}_coverage.png"),
        draw_coverage(sample, all_points, image_size, ratio),
    )
    cv2.imwrite(
        str(out_dir / f"{args.side}_undistort.png"),
        np.hstack((sample, cv2.undistort(sample, K, D))),
    )
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
