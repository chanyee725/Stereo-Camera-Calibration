"""Overlays drawn on preview and result images."""

import cv2

from utils.constants import FONT, FONT_SCALE, GRAY, GREEN, HLINE_SPACING, YELLOW


def draw_label(image, text: str, org=(10, 30), color=YELLOW, scale=FONT_SCALE):
    cv2.putText(image, text, org, FONT, scale, color, 2)


def draw_hlines(image, spacing=HLINE_SPACING, color=GREEN):
    h, w = image.shape[:2]
    for y in range(0, h, spacing):
        cv2.line(image, (0, y), (w, y), color, 1)


def draw_grid(image, grid, color=GRAY):
    h, w = image.shape[:2]
    gx, gy = grid
    for i in range(1, gx):
        cv2.line(image, (i * w // gx, 0), (i * w // gx, h), color, 1)
    for i in range(1, gy):
        cv2.line(image, (0, i * h // gy), (w, i * h // gy), color, 1)
