import cv2
import numpy as np
from threading import Lock

from flower_cbir.settings import load_config

_GRABCUT_LOCK = Lock()


def clean_mask(mask_binary):
    mask = np.asarray(mask_binary)
    if mask.ndim != 2 or mask.size == 0 or not np.isfinite(mask).all():
        raise ValueError('Mask must be a finite nonempty 2D array.')
    if not set(np.unique(mask)).issubset({0, 1, 255}):
        raise ValueError('Mask must be binary (0/1 or 0/255).')
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        raise ValueError('Mask has no foreground. Adjust the crop or GrabCut rectangle.')
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def run_grabcut(image_rgb: np.ndarray, rect=None, iter_count=None, mask_init: np.ndarray = None):
    """
    Run OpenCV GrabCut segmentation on a 384x384 RGB image.
    - If mask_init is provided: uses cv2.GC_INIT_WITH_MASK.
    - Otherwise: uses cv2.GC_INIT_WITH_RECT with specified bounding rect.
    Returns:
        mask_binary: uint8 (0 or 255) of largest connected foreground component.
        overlay: RGB uint8 image blending original image with foreground highlight.
        raw_mask: GrabCut internal mask (GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD).
    """
    config = load_config()['segmentation']
    rect = tuple(config['initial_rect']) if rect is None else rect
    iter_count = config['iter_count'] if iter_count is None else iter_count
    with _GRABCUT_LOCK:
        cv2.setRNGSeed(int(config['seed']))
        return _run_grabcut(image_rgb, rect, iter_count, mask_init)


def _run_grabcut(image_rgb, rect, iter_count, mask_init):
    h, w = image_rgb.shape[:2]
    if h < 3 or w < 3 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError('GrabCut requires an RGB image of at least 3x3 pixels.')
    # GrabCut requires BGR
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    if mask_init is not None:
        mask = mask_init.copy().astype(np.uint8)
        cv2.grabCut(img_bgr, mask, None, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_MASK)
    else:
        mask = np.zeros((h, w), np.uint8)
        # Ensure rect is within image boundaries
        rx, ry, rw, rh = rect
        rx = max(0, min(w - 2, rx))
        ry = max(0, min(h - 2, ry))
        rw = max(1, min(w - rx, rw))
        rh = max(1, min(h - ry, rh))
        cv2.grabCut(img_bgr, mask, (rx, ry, rw, rh), bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)

    # Foreground pixels are GC_FGD (1) and GC_PR_FGD (3)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    cleaned = clean_mask(fg_mask)

    # Create visualization overlay (tint foreground slightly with green/cyan or yellow border)
    overlay = image_rgb.copy()
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 128), 2)
    # Blend semi-transparent foreground highlight
    highlight = image_rgb.copy()
    highlight[cleaned == 0] = (highlight[cleaned == 0] * 0.35).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.6, highlight, 0.4, 0)

    return cleaned, overlay, mask


def create_mask_overlay(image_rgb: np.ndarray, mask_binary: np.ndarray) -> np.ndarray:
    """
    Create a clean overlay of binary mask on RGB image with green contour and darkened background.
    """
    overlay = image_rgb.copy()
    # Dim background
    bg_dimmed = (overlay * 0.3).astype(np.uint8)
    combined = np.where(mask_binary[:, :, None] > 0, overlay, bg_dimmed)
    # Draw contour
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(combined, contours, -1, (0, 255, 0), 2)
    return combined
