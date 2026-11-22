"""Executable provenance for the report-defined representation."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import PIL
import skimage

from flower_cbir.core.schema import EXTRACTOR_VERSION, FEATURE_NAMES
from flower_cbir.settings import PROJECT_ROOT, load_config


@lru_cache(maxsize=128)
def _cached_sha(path, size, modified_ns):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_sha256(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _cached_sha(str(path), stat.st_size, stat.st_mtime_ns)


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def extraction_contract():
    config = load_config()
    segmentation = config['segmentation']
    if segmentation['method'] not in ('deeplabv3', 'deeplabv3plus_v2'):
        raise ValueError('Unsupported segmentation method.')
    files = ['features.py', 'geometry.py', 'schema.py', 'segmentation.py',
             'preprocess.py', 'pipeline.py', 'deeplab.py', 'deeplab_worker.py',
             'normalization.py', 'contract.py']
    report = PROJECT_ROOT / 'flower_cbir/config/report_contract.json'
    report_info=json.loads(report.read_text(encoding='utf-8'))
    source_pdf=PROJECT_ROOT.parent/'doc'/report_info['source_pdf']
    if source_pdf.is_file() and file_sha256(source_pdf)!=report_info['source_sha256']:
        raise ValueError('The source report changed. Review the algorithms before creating a new extraction contract.')
    if segmentation['method'] == 'deeplabv3plus_v2':
        selected = {
            'method': 'deeplabv3plus_384_v2_finetuned',
            'checkpoint_sha256': file_sha256(segmentation['checkpoint']),
            'active_model_config_sha256': file_sha256(segmentation['active_model_config']),
            'postprocessing_sha256': file_sha256(segmentation['postprocessing_path']),
            'worker_sha256': file_sha256(segmentation['worker']),
            'model_loader_sha256': file_sha256(Path(segmentation['active_model_config']).with_suffix('.py')),
            'architecture_sha256': file_sha256(Path(segmentation['active_model_config']).with_name('train_flower_deeplabv3plus.py')),
            'postprocessing': segmentation['postprocessing'],
            'device': segmentation['device'], 'precision': 'float32',
            'input_size': 384, 'output_size': 384,
            'gallery_mask_source': config['dataset']['mask_source'],
        }
    else:
        selected = {'method': 'deeplabv3_resnet50',
                    'checkpoint_sha256': file_sha256(PROJECT_ROOT/segmentation['checkpoint']),
                    'device': segmentation['device'], 'precision': 'float32',
                    'normalization_mean': [.485, .456, .406],
                    'normalization_std': [.229, .224, .225],
                    'deterministic': True, 'decision': 'argmax_2classes'}
    return {
        'extractor_version': EXTRACTOR_VERSION, 'pipeline_version': config['pipeline_version'],
        'feature_names': FEATURE_NAMES, 'dtype': 'float32', 'image_size': config['image_size'],
        'epsilon': config['epsilon'], 'geometry': config['geometry'],
        'preprocessing': config['preprocessing'],
        'report': report_info,
        'source_sha256': {**{name: file_sha256(PROJECT_ROOT/'flower_cbir/core'/name) for name in files},
                          'settings.py':file_sha256(PROJECT_ROOT/'flower_cbir/settings.py')},
        'segmentation': selected,
        'runtime': {'numpy': np.__version__, 'opencv': cv2.__version__,
                    'skimage': skimage.__version__, 'pillow': PIL.__version__},
    }


def verify_contract(saved):
    current = extraction_contract()
    if fingerprint(current) != fingerprint(saved):
        raise ValueError('Index and query extraction contracts differ. Run python main.py --index to synchronize PostgreSQL.')
    return current
