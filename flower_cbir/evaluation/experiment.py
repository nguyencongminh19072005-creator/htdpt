import itertools
import json
import platform
import time
from collections import Counter

import cv2
import numpy as np

from flower_cbir.core.distance import compute_group_distances
from flower_cbir.core.normalization import SelectiveNormalizer
from flower_cbir.services.index_dataset import build_or_load_index
from flower_cbir.services.search import FlowerSearchEngine
from flower_cbir.settings import CONFIG_PATH, PROJECT_ROOT


def split_dataset(metadata, seed=42):
    rng = np.random.default_rng(seed)
    gallery, validation, test = [], [], []
    labels = np.array([item.get('label') or -1 for item in metadata])
    gallery.extend(np.flatnonzero(labels < 0).tolist())
    for label in sorted(set(labels[labels > 0])):
        indices = rng.permutation(np.flatnonzero(labels == label)).tolist()
        if len(indices) < 3:
            gallery.extend(indices[:1])
            test.extend(indices[1:])
        else:
            gallery_count = max(1, int(len(indices) * 0.6))
            validation_count = max(1, (len(indices) - gallery_count) // 2)
            gallery.extend(indices[:gallery_count])
            validation.extend(indices[gallery_count:gallery_count + validation_count])
            test.extend(indices[gallery_count + validation_count:])
    return np.array(sorted(gallery)), np.array(sorted(validation)), np.array(sorted(test))


def group_matrix(query_matrix, gallery_matrix):
    return np.stack([np.stack(compute_group_distances(query, gallery_matrix)[:4], axis=-1) for query in query_matrix])


def score(groups, weights, query_labels, gallery_labels, gallery_ids):
    distances = np.sum(groups * np.asarray(weights, dtype=np.float32), axis=-1)
    order = np.lexsort((np.broadcast_to(gallery_ids, distances.shape), distances), axis=1)[:, :5]
    relevant = gallery_labels[order] == query_labels[:, None]
    precision = relevant.mean(axis=1)
    available = np.sum(gallery_labels[None, :] == query_labels[:, None], axis=1)
    reciprocal = np.max(relevant / np.arange(1, 6), axis=1)
    return {'P_at_5': float(precision.mean()), 'macro_P_at_5': float(np.mean([precision[query_labels == label].mean() for label in np.unique(query_labels)])), 'Recall_at_5': float(np.mean(relevant.sum(axis=1) / np.maximum(available, 1))), 'Hit_at_5': float(relevant.any(axis=1).mean()), 'MRR_at_5': float(reciprocal.mean())}, precision, order


def run_experiment():
    output = PROJECT_ROOT / 'output/evaluation'
    output.mkdir(parents=True, exist_ok=True)
    data = build_or_load_index()
    metadata, raw = data['metadata'], data['raw_features']
    gallery, validation, test = split_dataset(metadata)
    normalizer = SelectiveNormalizer().fit(raw[gallery])
    normalized = normalizer.transform(raw)
    labels = np.array([item.get('label') or -1 for item in metadata])
    gallery_labels = labels[gallery]
    gallery_ids = np.array([metadata[index]['image_id'] for index in gallery])
    validation_groups = group_matrix(normalized[validation], normalized[gallery])
    test_groups = group_matrix(normalized[test], normalized[gallery])
    # No feature formula changes: tune only the unspecified group weights,
    # on validation, with all four groups retained and 0.05 grid resolution.
    candidates = [tuple(value / 20 for value in values) for values in itertools.product(range(1, 18), repeat=4) if sum(values) == 20]
    scored = [(score(validation_groups, weights, labels[validation], gallery_labels, gallery_ids)[0], weights) for weights in candidates]
    best_metrics, chosen = max(scored, key=lambda item: (item[0]['P_at_5'], item[0]['MRR_at_5'], -sum((value - 0.25) ** 2 for value in item[1])))
    configs = {'equal_weights': (0.25,) * 4, 'selected_weights': chosen, 'F1': (1, 0, 0, 0), 'F2': (0, 1, 0, 0), 'F3': (0, 0, 1, 0), 'F4': (0, 0, 0, 1), 'F1_F2': (0.5, 0.5, 0, 0), 'F3_F4': (0, 0, 0.5, 0.5), 'drop_F1': (0, 1/3, 1/3, 1/3), 'drop_F2': (1/3, 0, 1/3, 1/3), 'drop_F3': (1/3, 1/3, 0, 1/3), 'drop_F4': (1/3, 1/3, 1/3, 0)}
    results = {}
    for name, weights in configs.items():
        results[name] = {'weights': list(weights), 'validation': score(validation_groups, weights, labels[validation], gallery_labels, gallery_ids)[0], 'test': score(test_groups, weights, labels[test], gallery_labels, gallery_ids)[0]}
    _, baseline_precision, _ = score(test_groups, configs['equal_weights'], labels[test], gallery_labels, gallery_ids)
    _, selected_precision, selected_order = score(test_groups, chosen, labels[test], gallery_labels, gallery_ids)
    rng = np.random.default_rng(2026)
    difference = selected_precision - baseline_precision
    bootstrap = np.mean(difference[rng.integers(0, len(test), size=(2000, len(test)))], axis=1)
    split_ids = {name: [metadata[index]['image_id'] for index in indices] for name, indices in [('gallery', gallery), ('validation', validation), ('test', test)]}
    gallery_data = {**data, 'metadata': [metadata[index] for index in gallery], 'raw_features': raw[gallery], 'normalized_features': normalized[gallery], 'normalizer': normalizer}
    engine = FlowerSearchEngine(gallery_data)
    timings = []
    for iteration in range(105):
        _, elapsed = engine.search_by_vector(normalized[test[iteration % len(test)]], weights=chosen)
        if iteration >= 5:
            timings.append(elapsed)
    automatic = []
    for index in test[:20]:
        item = metadata[index]
        result = engine.search(item['image_path'], weights=chosen)
        relevant = [next(record for record in engine.metadata if record['image_id'] == hit['image_id'])['label'] == item['label'] for hit in result['results']]
        automatic.append({'query_id': item['image_id'], 'filename': item['filename'], 'P_at_5': float(np.mean(relevant)), 'timings': result['timings'], 'top5': result['results']})
    examples = []
    for row in [int(np.argmax(difference)), int(np.argmin(selected_precision)), 0]:
        index = int(test[row])
        results_list, _ = engine.search_by_vector(normalized[index], weights=chosen)
        examples.append({'query_id': metadata[index]['image_id'], 'filename': metadata[index]['filename'], 'label': int(labels[index]), 'P_at_5': float(selected_precision[row]), 'top5': results_list})
    report = {'index_version': data['version_id'], 'pipeline_version': data['manifest']['pipeline_version'], 'label_provenance': 'provisional_filename_mapping_not_source_verified', 'seed': 42, 'split_counts': {'gallery': len(gallery), 'validation': len(validation), 'test': len(test)}, 'split_ids': split_ids, 'classes': dict(sorted(Counter(labels[labels > 0].tolist()).items())), 'unknown_labels': [item['filename'] for item in metadata if item.get('label') is None], 'normalization_fit': 'gallery_only', 'candidate_count': len(candidates), 'selection': 'validation P@5, tie-break MRR@5 then proximity to equal weights; all four weights positive; step 0.05', 'selected_weights': list(chosen), 'results': results, 'paired_test_difference_bootstrap_95': np.percentile(bootstrap, [2.5, 97.5]).tolist(), 'benchmark': {'python': platform.python_version(), 'platform': platform.platform(), 'processor': platform.processor(), 'numpy': np.__version__, 'opencv': cv2.__version__, 'warmup': 5, 'repeats': 100, 'search_ms_p50': float(np.median(timings)), 'search_ms_p95': float(np.percentile(timings, 95))}, 'automatic_test_subset': automatic, 'automatic_test_subset_P_at_5': float(np.mean([item['P_at_5'] for item in automatic])), 'examples': examples, 'segmentation':data['manifest']['model_runtime'], 'limitations': ['Labels inferred from filenames; source provenance unverified.', 'Split by processed image, not by verified original photo group; near-duplicate leakage not excluded.', 'Relevance measures same species, not human-rated visual similarity.', 'The new segmentation heads were fine-tuned on gallery masks and selected on validation Dice; pretrained source overlap is unverified.', 'Fresh DeepLabV3 query features are checked separately for exact parity with saved masks.', 'Bootstrap is by query; does not account for class/source dependence.']}
    (output / 'experiment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'evaluation_normalizer.json').write_text(json.dumps(normalizer.to_dict(), indent=2), encoding='utf-8')
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    config['distance']['weights'] = {f'w{index + 1}': value for index, value in enumerate(chosen)}
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'split_counts': report['split_counts'], 'selected_weights': chosen, 'baseline_test': results['equal_weights']['test'], 'selected_test': results['selected_weights']['test'], 'automatic_P_at_5': report['automatic_test_subset_P_at_5'], 'benchmark': report['benchmark']}, indent=2))


if __name__ == '__main__':
    run_experiment()
