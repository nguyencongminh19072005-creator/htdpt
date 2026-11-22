import cv2
import numpy as np
from PIL import Image, ImageOps
import io
from pathlib import Path


def square_crop_bounds(flower_box):
    """Report 1.1.3: center on flower box, square side 1.10*max(width,height)."""
    values = np.asarray(flower_box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values[2:] <= 0):
        raise ValueError('Flower bounding box must be finite (x,y,width,height) with positive size.')
    x, y, width, height = values
    side = int(np.ceil(max(width, height) * 11.0 / 10.0))
    left = int(np.floor(x + width/2 - side/2))
    top = int(np.floor(y + height/2 - side/2))
    return [left, top, side, side]


def crop_with_padding(image, bounds, fill):
    left, top, width, height = bounds
    shape = (height, width) + image.shape[2:]
    crop = np.full(shape, fill, dtype=image.dtype)
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(image.shape[1], left+width), min(image.shape[0], top+height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('The flower bounding box does not intersect the source image.')
    crop[y0-top:y1-top, x0-left:x1-left] = image[y0:y1, x0:x1]
    return crop


def load_image(input_source):
    """
    Load image from file path, Path object, raw bytes, or PIL Image.
    Handles EXIF orientation and ensures RGB format.
    """
    if isinstance(input_source, (str, Path)):
        pil_img = Image.open(str(input_source))
    elif isinstance(input_source, bytes):
        pil_img = Image.open(io.BytesIO(input_source))
    elif isinstance(input_source, io.BytesIO):
        pil_img = Image.open(input_source)
    elif isinstance(input_source, Image.Image):
        pil_img = input_source
    elif isinstance(input_source, np.ndarray):
        if input_source.size == 0 or input_source.dtype != np.uint8:
            raise ValueError('Array input must be a nonempty uint8 RGB image.')
        if input_source.ndim == 2:
            return np.stack([input_source] * 3, axis=-1).astype(np.uint8)
        if input_source.ndim != 3 or input_source.shape[2] not in (3, 4):
            raise ValueError('Array input requires 3 RGB or 4 RGBA channels.')
        pil_img = Image.fromarray(input_source)
    else:
        raise ValueError(f"Unsupported image input type: {type(input_source)}")

    # EXIF orientation correction
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    # Convert to RGB (handle RGBA / Grayscale / Palette)
    if pil_img.mode == 'RGBA':
        # Composite on white background as specified in guide
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg
    else:
        pil_img = pil_img.convert("RGB")

    return np.array(pil_img, dtype=np.uint8)


def resize_image(image_rgb: np.ndarray, target_size=(384, 384)) -> np.ndarray:
    """
    Resize RGB image using cv2.INTER_AREA for downscaling and cv2.INTER_CUBIC for upscaling.
    """
    h, w = image_rgb.shape[:2]
    tw, th = target_size
    if (w, h) == (tw, th):
        return image_rgb

    if tw <= w and th <= h:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    return cv2.resize(image_rgb, (tw, th), interpolation=interpolation)


def square_center_crop(image_rgb: np.ndarray) -> np.ndarray:
    """
    Square crop keeping center.
    """
    h, w = image_rgb.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return image_rgb[top:top + min_dim, left:left + min_dim]


def preprocess_query_image(input_source, target_size=(384, 384), crop_box=None) -> np.ndarray:
    """
    Full preprocessing pipeline for a query image:
    1. Load + EXIF fix + RGB conversion.
    2. Optional crop (custom crop_box=(x, y, w, h) or square center crop).
    3. Resize to target_size (default 384x384).
    """
    img = load_image(input_source)
    if crop_box is not None:
        x, y, w, h = crop_box
        ih, iw = img.shape[:2]
        x = max(0, min(iw - 1, int(x)))
        y = max(0, min(ih - 1, int(y)))
        w = max(1, min(iw - x, int(w)))
        h = max(1, min(ih - y, int(h)))
        img = img[y:y + h, x:x + w]
    else:
        img = square_center_crop(img)

    return resize_image(img, target_size)


def preprocess_mask(mask, source_shape, target_size=(384, 384), crop_box=None):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != tuple(source_shape):
        raise ValueError('Supplied mask must match the EXIF-oriented source image shape.')
    height, width = source_shape
    if crop_box is None:
        side = min(height, width)
        left, top = (width - side) // 2, (height - side) // 2
        mask = mask[top:top + side, left:left + side]
    else:
        left, top, crop_width, crop_height = crop_box
        left, top = max(0, min(width - 1, int(left))), max(0, min(height - 1, int(top)))
        crop_width = max(1, min(width - left, int(crop_width)))
        crop_height = max(1, min(height - top, int(crop_height)))
        mask = mask[top:top + crop_height, left:left + crop_width]
    return cv2.resize(mask.astype(np.uint8), tuple(target_size), interpolation=cv2.INTER_NEAREST)
