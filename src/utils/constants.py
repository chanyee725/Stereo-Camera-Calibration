"""Constants shared across the scripts. User-tunable values live in configs/."""

from pathlib import Path

import cv2

# Paths
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
OUTPUT_ROOT = ROOT / "outputs"

# Session layout
IMAGE_DIRS = ("left", "right")
# Left and right are numbered independently, so stereo pairs are recorded here
PAIRS_FILE = "pairs.csv"
CALIB_DIR = "calib"

# Quality checks
COVERAGE_GRID = (8, 6)
# Views whose error exceeds this multiple of the median are flagged
OUTLIER_FACTOR = 2

# Drawing (BGR)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GRAY = (128, 128, 128)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
# Spacing in pixels of the horizontal lines drawn on rectified images
HLINE_SPACING = 40
