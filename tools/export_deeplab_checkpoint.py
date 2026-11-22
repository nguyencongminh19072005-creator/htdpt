"""Export a local, trusted training checkpoint to inference-only weights."""
import argparse
import hashlib
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    payload = torch.load(args.source, map_location='cpu', weights_only=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': payload['model'], 'architecture': 'deeplabv3_resnet50_2classes_aux',
                'training_source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
                'epoch': int(payload['epoch']), 'best_validation_dice': float(payload['best_metric'])}, args.output)
    print(args.output, args.output.stat().st_size, hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
