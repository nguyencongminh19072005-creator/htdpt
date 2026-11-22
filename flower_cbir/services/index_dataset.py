"""Immutable artifacts built with the query pipeline; database publishing is explicit."""
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from flower_cbir.core.contract import extraction_contract, file_sha256, fingerprint, verify_contract
from flower_cbir.core.deeplab import get_segmenter
from flower_cbir.core.normalization import SelectiveNormalizer
from flower_cbir.core.pipeline import extract_image
from flower_cbir.core.preprocess import load_image
from flower_cbir.core.schema import validate_features
from flower_cbir.core.segmentation import clean_mask
from flower_cbir.settings import PROJECT_ROOT, load_config

file_hash = file_sha256


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def find_dataset_paths():
    dataset = load_config()['dataset']
    return ((PROJECT_ROOT / dataset['image_dir']).resolve(),
            (PROJECT_ROOT / dataset['mask_dir']).resolve())


def _load_version(directory, output_dir, expected_fingerprint=None):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    verify_contract(manifest['contract'])
    if expected_fingerprint is not None and manifest['fingerprint'] != expected_fingerprint:
        raise ValueError('Index fingerprint does not match its source dataset.')
    for name, expected in manifest['artifact_hashes'].items():
        if file_sha256(directory/name) != expected:
            raise ValueError(f'Index checksum mismatch: {name}')
    raw = np.load(directory/'features_raw.npy', allow_pickle=False)
    normalized = np.load(directory/'features_normalized.npy', allow_pickle=False)
    metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    normalizer = SelectiveNormalizer.from_dict(json.loads((directory/'normalizer.json').read_text(encoding='utf-8')))
    validate_features(raw)
    if raw.shape != (len(metadata), 163) or not np.array_equal(normalizer.transform(raw), normalized):
        raise ValueError('Saved features, metadata and normalizer disagree.')
    return {'raw_features': raw, 'normalized_features': normalized, 'metadata': metadata,
            'normalizer': normalizer, 'output_dir': Path(output_dir), 'version_dir': directory,
            'version_id': manifest['version_id'], 'manifest': manifest, 'contract': manifest['contract']}


def build_or_load_index(output_dir=None, force_recompute=False, image_dir=None,
                        mask_dir=None, expected_count=None):
    """Use the selected DeepLabV3+ and postprocessing for gallery and queries.

    Existing RGB 384x384 images retain their published framing. Source images
    are never modified. All tools call extract_image for the feature vector.
    """
    config = load_config()
    output_dir = Path(output_dir) if output_dir else PROJECT_ROOT/'flower_cbir/data'
    image_dir = Path(image_dir).resolve() if image_dir else find_dataset_paths()[0]
    mask_dir = Path(mask_dir).resolve() if mask_dir else None
    expected_count = config['dataset']['expected_count'] if expected_count is None else expected_count
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not images or (expected_count and len(images) != expected_count):
        raise ValueError(f'Expected {expected_count} source images, found {len(images)}.')
    if len({p.stem for p in images}) != len(images):
        raise ValueError('Image stems must be unique across file formats.')
    contract = extraction_contract()
    segmenter = get_segmenter()
    model_info = segmenter.info
    segmentation_key = fingerprint({'model': model_info,
        'worker': contract['segmentation'].get('worker_sha256',
            contract['source_sha256']['deeplab_worker.py']),
        'clean_mask': contract['source_sha256']['segmentation.py']})[:20]
    mask_root = PROJECT_ROOT/config['dataset']['mask_dir']/segmentation_key
    reviewed = config['dataset'].get('mask_source') == 'reviewed_final_segmentation_v2'
    reviewed_dir = Path(config['dataset']['reviewed_mask_dir']).resolve() if reviewed else None
    labels_path = (PROJECT_ROOT/config['dataset']['labels_mat']).resolve()
    labels = loadmat(labels_path)['labels'].ravel() if labels_path.is_file() else []
    metadata, timings = [], []
    for position, image_path in enumerate(images, 1):
        with Image.open(image_path) as original_image:
            if original_image.size!=(384,384) or image_path.suffix.lower()!='.png':
                raise ValueError(f'{image_path.name}: catalog inputs must be PNG 384x384.')
        rgb = load_image(image_path)
        if rgb.shape != (384, 384, 3):
            raise ValueError(f'{image_path.name}: batch images must already be RGB 384x384 as in the report.')
        image_hash = file_sha256(image_path)
        pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        cached_mask = (reviewed_dir/(image_path.stem+'.png') if reviewed else
                       mask_root/pixel_hash[:20]/(image_path.stem+'.png'))
        cached_note = cached_mask.with_suffix('.json')
        mask = None
        if reviewed:
            if not cached_mask.is_file():
                raise FileNotFoundError(f'Missing reviewed v2 mask: {cached_mask}')
            mask = clean_mask(np.asarray(Image.open(cached_mask).convert('L')))
        elif cached_mask.is_file() and cached_note.is_file():
            note = json.loads(cached_note.read_text(encoding='utf-8'))
            if note.get('pixel_sha256') == pixel_hash and note.get('mask_sha256') == file_sha256(cached_mask):
                mask = clean_mask(np.asarray(Image.open(cached_mask).convert('L')))
        start = time.perf_counter()
        if mask is None:
            mask = segmenter.predict(rgb)
            cached_mask.parent.mkdir(parents=True, exist_ok=True)
            temporary = cached_mask.with_name(cached_mask.stem+'.tmp.png')
            Image.fromarray(mask).save(temporary)
            os.replace(temporary, cached_mask)
            atomic_json(cached_note, {'pixel_sha256': pixel_hash, 'mask_sha256': file_sha256(cached_mask),
                                     'model': model_info, 'segmentation_key': segmentation_key})
        if mask_dir is not None:
            supplied = clean_mask(np.asarray(Image.open(mask_dir/(image_path.stem+'.png')).convert('L')))
            if not np.array_equal(mask, supplied):
                raise ValueError(f'{image_path.name}: supplied mask differs from this DeepLabV3 pipeline. '
                                 'Omit --mask-dir to generate consistent masks.')
        timings.append((time.perf_counter()-start)*1000)
        try:
            source_id = int(image_path.stem.split('_')[-1])
        except ValueError:
            source_id = position
        metadata.append({'image_id': source_id, 'source_id': source_id, 'filename': image_path.name,
            'image_path': str(image_path), 'mask_path': str(cached_mask),
            'image_sha256': image_hash, 'mask_sha256': file_sha256(cached_mask),
            'pixel_sha256': pixel_hash, 'mask_provenance': config['dataset']['mask_source'],
            'segmentation_key': segmentation_key,
            'label': int(labels[source_id-1]) if 1 <= source_id <= len(labels) else None,
            'label_provenance': config['dataset']['label_provenance']})
        if position % 100 == 0 or position == len(images):
            print(f'DeepLabV3+ postprocessed masks verified: {position}/{len(images)}', flush=True)
    signature = {'records': metadata, 'contract': contract, 'model_runtime': model_info,
                 'labels_sha256': file_sha256(labels_path) if labels_path.is_file() else None}
    identity = fingerprint(signature)
    version_id = contract['extractor_version']+'-'+identity[:20]
    directory = output_dir/'versions'/version_id
    if (directory/'manifest.json').is_file() and not force_recompute:
        return _load_version(directory, output_dir, identity)
    original = _load_version(directory, output_dir, identity) if (directory/'manifest.json').is_file() else None
    raw_vectors, areas = [], []
    started = time.perf_counter()
    for position, item in enumerate(metadata, 1):
        mask = np.asarray(Image.open(item['mask_path']).convert('L'))
        extracted = extract_image(item['image_path'], supplied_mask=mask)
        raw_vectors.append(extracted['raw_vector'])
        areas.append(float((extracted['mask_binary']>0).mean()))
        if position % 100 == 0 or position == len(metadata):
            print(f'Shared 163D extractor: {position}/{len(metadata)}', flush=True)
    raw = np.asarray(raw_vectors, dtype=np.float32)
    validate_features(raw)
    if original is not None:
        if not np.array_equal(original['raw_features'], raw):
            raise ValueError('Re-extraction differs from the immutable index with the same signature.')
        return original
    directory.mkdir(parents=True, exist_ok=True)
    normalizer = SelectiveNormalizer(epsilon=config['epsilon']).fit(raw)
    normalized = normalizer.transform(raw)
    np.save(directory/'features_raw.npy', raw)
    np.save(directory/'features_normalized.npy', normalized)
    atomic_json(directory/'metadata.json', metadata)
    atomic_json(directory/'normalizer.json', normalizer.to_dict())
    elapsed = time.perf_counter()-started
    summary = {'total_images': len(metadata), 'feature_dims': 163, 'failed_count': 0,
               'extraction_time_s': elapsed, 'avg_ms_per_image': elapsed*1000/len(metadata),
               'segmentation_ms_sum': sum(timings),
               'area_fraction_percentiles': np.percentile(areas,[0,25,50,75,100]).tolist()}
    manifest = {'version_id': version_id, 'fingerprint': identity,
        'pipeline_version': config['pipeline_version'], 'contract': contract,
        'model_runtime': model_info, 'signature': signature, 'summary': summary,
        'artifact_hashes': {name: file_sha256(directory/name) for name in
            ['features_raw.npy','features_normalized.npy','metadata.json','normalizer.json']}}
    atomic_json(directory/'manifest.json', manifest)
    atomic_json(output_dir/'active_index.json', {'fingerprint': identity,
        'directory': directory.relative_to(output_dir).as_posix(), 'version_id': version_id})
    return _load_version(directory, output_dir, identity)
