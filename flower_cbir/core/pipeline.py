"""The only image -> mask -> 163D entry point, for batch import and queries."""
import time

import cv2
import numpy as np

from flower_cbir.core.deeplab import run_deeplab
from flower_cbir.core.features import extract_all_features
from flower_cbir.core.preprocess import load_image, resize_image, square_crop_bounds, crop_with_padding
from flower_cbir.core.segmentation import clean_mask, create_mask_overlay
from flower_cbir.settings import load_config


def prepare_image(source, supplied_mask=None, flower_box=None, crop_box=None):
    """Prepared 384x384 images are preserved. New framing follows report section 1.1.3.

    A user flower bounding box is expanded by 1.10. For a nonprepared image with
    no box, a coarse DeepLab pass estimates that box; the final mask is always
    predicted again on the final RGB crop. No mask is used to paint image pixels.
    """
    image = load_image(source)
    height, width = image.shape[:2]
    mask = None if supplied_mask is None else np.asarray(supplied_mask)
    if mask is not None and (mask.ndim != 2 or mask.shape != (height, width)):
        raise ValueError('The supplied mask must match the EXIF-oriented source image.')
    if mask is not None:
        mask = clean_mask(mask)
    bounds = None
    if flower_box is not None:
        bounds = square_crop_bounds(flower_box)
    elif crop_box is not None:
        # A manually supplied box denotes the flower bounding box, not a second crop policy.
        bounds = square_crop_bounds(crop_box)
    elif (width, height) != (384, 384):
        if mask is not None:
            ys, xs = np.nonzero(mask)
            flower_box = (int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
        else:
            provisional = resize_image(image)
            coarse, _, _ = run_deeplab(provisional)
            ys, xs = np.nonzero(coarse)
            flower_box = (xs.min()*width/384, ys.min()*height/384,
                          (xs.max()-xs.min()+1)*width/384, (ys.max()-ys.min()+1)*height/384)
        bounds = square_crop_bounds(flower_box)
    if bounds is not None:
        image = crop_with_padding(image, bounds, fill=128)
        if mask is not None:
            mask = crop_with_padding(mask, bounds, fill=0)
    image = resize_image(image)
    if mask is not None:
        mask = clean_mask(cv2.resize(mask, (384, 384), interpolation=cv2.INTER_NEAREST))
    return image, mask, {'source_size': [width, height], 'crop_bounds': bounds,
                         'crop_margin': 1.10, 'padding_rgb': [128, 128, 128]}


def extract_image(source, supplied_mask=None, flower_box=None, crop_box=None):
    timings = {}
    started = time.perf_counter()
    rgb, mask, preparation = prepare_image(source, supplied_mask, flower_box, crop_box)
    timings['preprocess_ms'] = (time.perf_counter()-started)*1000
    started = time.perf_counter()
    if mask is None:
        mask, overlay, model_info = run_deeplab(rgb)
        segmentation = {'method': load_config()['segmentation']['method'], **model_info}
    else:
        overlay = create_mask_overlay(rgb, mask)
        segmentation = {'method': 'supplied_mask'}
    timings['segmentation_ms'] = (time.perf_counter()-started)*1000
    started = time.perf_counter()
    vector, explanation = extract_all_features(rgb, mask)
    timings['extraction_ms'] = (time.perf_counter()-started)*1000
    return {'image_rgb': rgb, 'mask_binary': mask, 'overlay': overlay,
            'raw_vector': vector, 'explanation': explanation,
            'preparation': preparation, 'segmentation': segmentation, 'timings': timings}
