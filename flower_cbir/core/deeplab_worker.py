"""Persistent DeepLabV3 inference worker; only this process imports PyTorch.

JSON lines carry base64 uint8 RGB and label bytes, never paths or pickle objects.
This lets the feature application use a different Python from the GPU environment.
"""
import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torchvision
    from torchvision.models.segmentation import deeplabv3_resnet50

    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA device is unavailable. Select cpu and rebuild the index.')
    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=2, aux_loss=True)
    model.load_state_dict(payload['model'], strict=True)
    model.to(args.device).eval()
    mean = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
    std = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
    print(json.dumps({'ready': True, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                      'device': args.device, 'precision': 'float32', 'deterministic': True,
                      'cuda': str(torch.version.cuda), 'cudnn': torch.backends.cudnn.version(),
                      'gpu_name': torch.cuda.get_device_name(0) if args.device=='cuda' else None}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('action') == 'close':
                break
            raw = base64.b64decode(request['rgb'], validate=True)
            if len(raw) != 384 * 384 * 3:
                raise ValueError('DeepLabV3 requires exactly 384x384 uint8 RGB.')
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(384, 384, 3)
            array = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
            tensor = torch.from_numpy(((array - mean) / std).copy()).unsqueeze(0).to(args.device)
            with torch.inference_mode():
                mask = model(tensor)['out'].argmax(1)[0].cpu().numpy().astype(np.uint8)
            response = {'mask': base64.b64encode(mask.tobytes()).decode('ascii')}
        except Exception as exc:
            response = {'error': f'{type(exc).__name__}: {exc}'}
        print(json.dumps(response), flush=True)


if __name__ == '__main__':
    try:
        run()
    except Exception as exc:
        print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}), flush=True)
        raise
