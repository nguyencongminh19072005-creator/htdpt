"""Evaluate 384-pixel DeepLab inference settings on the fixed reviewed split.

Threshold and test-time augmentation are selected on validation only. The
held-out split is measured once after selection and never used for tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import refit_deeplab_corrected_v2 as refit
from train_flower_deeplabv3plus import DeepLabV3Plus


def probability(model: DeepLabV3Plus, rgb: np.ndarray, rotations: int) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for k in range(rotations):
            rotated = np.ascontiguousarray(np.rot90(rgb, k))
            x = torch.from_numpy(rotated.transpose(2, 0, 1)[None]).float() / 255
            output = model(x).sigmoid()[0, 0].numpy()
            outputs.append(np.rot90(output, -k))
    return np.mean(outputs, axis=0)


def score(probabilities: dict[str, np.ndarray], truths: dict[str, np.ndarray],
          threshold: float) -> dict:
    rows = []
    for name, prob in probabilities.items():
        pred = prob >= threshold
        truth = truths[name]
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        rows.append({'filename': name, 'iou': tp / max(tp + fp + fn, 1),
                     'precision': tp / max(tp + fp, 1),
                     'recall': tp / max(tp + fn, 1)})
    return {'mean_iou': float(np.mean([r['iou'] for r in rows])),
            'mean_precision': float(np.mean([r['precision'] for r in rows])),
            'mean_recall': float(np.mean([r['recall'] for r in rows])),
            'per_image': rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path, nargs='+')
    args = parser.parse_args()
    _, groups, _ = refit.prepare()
    models = []
    for checkpoint in args.checkpoints:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if int(state['size']) != 384:
            raise ValueError('A 384x384 checkpoint is required')
        model = DeepLabV3Plus(width=int(state['width']), rates=tuple(state['rates']))
        model.load_state_dict(state['state_dict'])
        model.eval()
        models.append(model)
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    names = groups['validation'] + groups['reviewed_test_current']
    truths = {name: cv2.imread(str(refit.MASKS / name), cv2.IMREAD_GRAYSCALE) > 127
              for name in names}
    images = {name: cv2.cvtColor(cv2.imread(str(refit.IMAGES / name)), cv2.COLOR_BGR2RGB)
              for name in names}
    probabilities = {count: {name: np.mean([probability(model, images[name], count)
                                            for model in models], axis=0)
                             for name in names} for count in (1, 4)}
    thresholds = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
    validation = {}
    for count in (1, 4):
        subset = {name: probabilities[count][name] for name in groups['validation']}
        validation[str(count)] = {str(t): score(subset, truths, t) for t in thresholds}
    choices = [(validation[str(count)][str(t)]['mean_iou'],
                -abs(t - 0.5), -count, count, t)
               for count in (1, 4) for t in thresholds]
    _, _, _, count, threshold = max(choices)
    heldout = score({name: probabilities[count][name]
                     for name in groups['reviewed_test_current']}, truths, threshold)
    baseline = score({name: probabilities[1][name]
                      for name in groups['reviewed_test_current']}, truths, 0.5)
    result = {'checkpoints': [str(path.resolve()) for path in args.checkpoints],
              'input_size': 384,
              'validation_count': len(groups['validation']),
              'heldout_count': len(groups['reviewed_test_current']),
              'validation_grid': validation,
              'selected_on_validation': {'rotations': count, 'threshold': threshold,
                                          **validation[str(count)][str(threshold)]},
              'heldout_selected': heldout, 'heldout_baseline': baseline}
    suffix = '_ensemble' if len(args.checkpoints) > 1 else ''
    destination = args.checkpoints[0].parent / f'inference_tuning{suffix}.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                           encoding='utf-8')
    print(json.dumps({'selected_rotations': count, 'selected_threshold': threshold,
                      'validation_iou': validation[str(count)][str(threshold)]['mean_iou'],
                      'heldout_iou_selected': heldout['mean_iou'],
                      'heldout_iou_baseline': baseline['mean_iou'],
                      'report': str(destination)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
