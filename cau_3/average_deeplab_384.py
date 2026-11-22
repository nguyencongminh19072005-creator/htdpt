"""Create a single 384x384 checkpoint by averaging compatible model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from refit_deeplab_corrected_v2 import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--second-weight', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.second_weight < 1:
        parser.error('--second-weight must be between zero and one')
    if args.destination.exists():
        raise FileExistsError(args.destination)
    states = [torch.load(path, map_location='cpu', weights_only=True)
              for path in (args.first, args.second)]
    if any(int(state['size']) != 384 for state in states):
        raise ValueError('Both checkpoints must use input 384x384')
    if any(states[0][key] != states[1][key] for key in ('width', 'rates')):
        raise ValueError('Architecture mismatch')
    weights = states[0]['state_dict']
    if weights.keys() != states[1]['state_dict'].keys():
        raise ValueError('State dictionary keys differ')
    mix = {}
    for key, first in weights.items():
        second = states[1]['state_dict'][key]
        if first.shape != second.shape:
            raise ValueError(f'Shape mismatch for {key}')
        mix[key] = ((1 - args.second_weight) * first + args.second_weight * second
                    if first.is_floating_point() else first.clone())
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': mix, 'size': 384, 'width': states[0]['width'],
                'rates': states[0]['rates'], 'architecture': 'compact_deeplabv3plus_weight_average',
                'parents': [str(args.first.resolve()), str(args.second.resolve())],
                'parent_sha256': [sha256(args.first), sha256(args.second)],
                'second_weight': args.second_weight}, args.destination)
    print(json.dumps({'checkpoint': str(args.destination.resolve()),
                      'sha256': sha256(args.destination)}))


if __name__ == '__main__':
    main()
