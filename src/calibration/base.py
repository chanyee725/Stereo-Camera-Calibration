"""Base class holding the steps shared by every calibration."""

from abc import ABC, abstractmethod
from pathlib import Path

import cv2

from utils.board import build_board, detect
from utils.constants import CALIB_DIR, OUTLIER_FACTOR


class Calibration(ABC):
    """Shared board setup and output handling; subclasses implement the steps."""

    def __init__(self, cfg: dict, session: Path):
        self.cfg = cfg
        self.session = session
        self.out_dir = session / CALIB_DIR
        self.min_corners = cfg.get("min_corners", 6)
        self.board = build_board(cfg)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.image_size = None

    def run(self):
        self.collect()
        self.calibrate()
        self.report()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.save()

    @abstractmethod
    def collect(self):
        """Detect corners and gather the views used for calibration."""

    @abstractmethod
    def calibrate(self):
        """Run the OpenCV calibration on the collected views."""

    @abstractmethod
    def report(self):
        """Print the results and quality metrics."""

    @abstractmethod
    def save(self):
        """Write the results to self.out_dir."""

    def detect(self, image) -> dict:
        return detect(self.detector, image)

    def write_yaml(self, name: str, values: dict):
        fs = cv2.FileStorage(str(self.out_dir / name), cv2.FILE_STORAGE_WRITE)
        fs.write("image_width", self.image_size[0])
        fs.write("image_height", self.image_size[1])
        for key, value in values.items():
            if isinstance(value, list):
                fs.startWriteStruct(key, cv2.FileNode_SEQ)
                for item in value:
                    fs.write("", item)
                fs.endWriteStruct()
            else:
                fs.write(key, value)
        fs.release()

    @staticmethod
    def outlier_flag(error: float, median: float) -> str:
        return "  <-- check" if error > OUTLIER_FACTOR * median else ""
