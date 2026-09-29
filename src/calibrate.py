"""ChArUco calibration of a capture session recorded by acquire_image.py.

    mono    calibrate one camera (run for --side left and --side right first)
    stereo  estimate the relative pose using the fixed mono intrinsics

Results are written to <session>/calib/.
"""

import argparse
from pathlib import Path

from calibration import MonoCalibration, StereoCalibration
from utils.config import load_config
from utils.constants import DEFAULT_CONFIG, IMAGE_DIRS
from utils.session import latest_session


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    common.add_argument(
        "--session", type=Path, help="Capture directory (default: latest in outputs/)"
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    mono = sub.add_parser("mono", parents=[common], help="Single camera intrinsics")
    mono.add_argument("--side", required=True, choices=IMAGE_DIRS)
    mono.add_argument(
        "--images",
        type=Path,
        nargs="+",
        help="Image directories (default: <session>/<side>)",
    )
    stereo = sub.add_parser("stereo", parents=[common], help="Stereo extrinsics")
    stereo.add_argument(
        "--pair",
        help="Left image name of the pair for rectified_check.png (default: first)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)["calibration"]
    session = args.session or latest_session()
    if args.mode == "mono":
        calibration = MonoCalibration(cfg, session, args.side, args.images)
    else:
        calibration = StereoCalibration(cfg, session, args.pair)
    calibration.run()


if __name__ == "__main__":
    main()
