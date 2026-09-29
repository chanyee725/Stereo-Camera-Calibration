"""Live preview of the stereo cameras and capture of calibration images.

Keys:
    s: save the current left/right pair  -> <session>/left, right (+ pairs.csv)
    l: save the left image only          -> <session>/left
    r: save the right image only         -> <session>/right
    q: quit

Each folder is numbered on its own (last index + 1), so pair names can differ.
Stereo pairs saved with 's' are recorded in <session>/pairs.csv.

The board is detected at save time: an image with fewer than min_corners corners
is not saved, and a pair is recorded only if both images share enough corners.

A new session is created on every run. Use --resume to add images to the latest
session, or --resume <session> to add to a specific one.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from utils.board import build_board, detect
from utils.config import load_config
from utils.constants import DEFAULT_CONFIG, GREEN, IMAGE_DIRS, OUTPUT_ROOT, RED
from utils.draw import draw_label
from utils.session import append_pair, last_index, latest_session, save_image

WINDOW_NAME = "stereo"


def open_capture(device, width: int, height: int, fps: int, fourcc: str):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {device}")
    # Some UVC drivers require FOURCC to be set before the resolution
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


class StereoCamera:
    def __init__(self, cfg: dict):
        self.mode = cfg["mode"]
        self.swap_lr = cfg.get("swap_lr", False)
        self.caps = []

        width, height = cfg["width"], cfg["height"]
        fps, fourcc = cfg["fps"], cfg["fourcc"]
        try:
            if self.mode == "dual":
                for key in ("left_device", "right_device"):
                    self.caps.append(open_capture(cfg[key], width, height, fps, fourcc))
            elif self.mode == "side_by_side":
                self.caps.append(
                    open_capture(cfg["device"], 2 * width, height, fps, fourcc)
                )
            else:
                raise ValueError(
                    f"Unsupported camera mode for live capture: {self.mode}"
                )
        except Exception:
            self.release()
            raise

    def read(self):
        if self.mode == "dual":
            # Grab both first, then decode, to keep the two frames close in time
            if not all(cap.grab() for cap in self.caps):
                return None, None
            frames = [cap.retrieve()[1] for cap in self.caps]
            left, right = frames
        else:
            ok, frame = self.caps[0].read()
            if not ok:
                return None, None
            half = frame.shape[1] // 2
            left, right = frame[:, :half], frame[:, half:]

        if left is None or right is None:
            return None, None
        if self.swap_lr:
            left, right = right, left
        return left, right

    def release(self):
        for cap in self.caps:
            cap.release()
        self.caps = []


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--resume",
        type=Path,
        nargs="?",
        const="latest",
        help="Continue an existing session (default: latest in outputs/)",
    )
    args = parser.parse_args()

    if args.resume is None:
        session_dir = OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d:%H-%M-%S")
    elif str(args.resume) == "latest":
        session_dir = latest_session()
    else:
        session_dir = args.resume
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Session not found: {session_dir}")

    # Continue numbering after the images already in the session
    counts = {side: last_index(session_dir / side) for side in IMAGE_DIRS}
    action = "Resuming" if args.resume else "New"
    print(f"{action} session: {session_dir} (last index: {counts})")

    cfg = load_config(args.config)
    detector = cv2.aruco.CharucoDetector(build_board(cfg["calibration"]))
    min_corners = cfg["calibration"].get("min_corners", 6)
    camera = StereoCamera(cfg["camera"])
    status, status_color = "", GREEN

    try:
        while True:
            left, right = camera.read()
            if left is None:
                print("Failed to read frames from the cameras")
                break

            if right.shape != left.shape:
                right_view = cv2.resize(right, (left.shape[1], left.shape[0]))
            else:
                right_view = right
            preview = np.hstack((left, right_view))
            draw_label(
                preview,
                f"left: {counts['left']}  right: {counts['right']}  "
                "[s] pair [l] left [r] right [q] quit",
                color=GREEN,
            )
            if status:
                draw_label(preview, status, (10, 60), status_color, scale=0.6)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            targets = {
                ord("s"): (("left", left), ("right", right)),
                ord("l"): (("left", left),),
                ord("r"): (("right", right),),
            }.get(key)
            if targets:
                detections = {side: detect(detector, image) for side, image in targets}
                saved, messages, ok = {}, [], True
                for side, image in targets:
                    found = len(detections[side])
                    if found < min_corners:
                        messages.append(f"{side} skipped ({found} corners)")
                        ok = False
                        continue
                    counts[side] += 1
                    saved[side] = save_image(session_dir / side, counts[side], image)
                    messages.append(f"{side}/{saved[side]} ({found} corners)")
                if len(targets) == 2:
                    common = len(set(detections["left"]) & set(detections["right"]))
                    if len(saved) == 2 and common >= min_corners:
                        append_pair(session_dir, saved["left"], saved["right"])
                        messages.append(f"pair recorded ({common} common)")
                    else:
                        messages.append(f"pair not recorded ({common} common)")
                        ok = False
                status = ", ".join(messages)
                status_color = GREEN if ok else RED
                print(status)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
