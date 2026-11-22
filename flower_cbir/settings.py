import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / 'flower_cbir/config/pipeline.json'


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if config['image_size'] != [384, 384] or config['features']['total_dims'] != 163:
        raise ValueError('Pipeline requires 384x384 images and the 163D feature schema.')
    if config['geometry'] != {'b5_kernel_shape': 'ellipse', 'b5_kernel_size': [5, 5], 'erode_iterations': 1}:
        raise ValueError('The report fixes erosion to one 5x5 ellipse with zero outside the image.')
    if config['epsilon'] != 1e-12:
        raise ValueError('The report fixes epsilon to 1e-12.')
    if config.get('preprocessing') != {'crop_margin':1.10,'padding_rgb':[128,128,128],'prepared_size':[384,384]}:
        raise ValueError('The report fixes square crop margin to 1.10, neutral gray padding, RGB 384x384.')
    validate_weights(tuple(config['distance']['weights'].values()))
    return config


def validate_weights(weights):
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError('Weights must contain four finite nonnegative values with positive sum.')
    return tuple((values / values.sum()).tolist())


def default_weights():
    config = load_config()
    return validate_weights([config['distance']['weights'][f'w{index}'] for index in range(1, 5)])
