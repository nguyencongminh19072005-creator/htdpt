"""One deterministic segmentation interface shared by the query and import tools."""
import atexit
import base64
import json
import os
import queue
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from flower_cbir.core.segmentation import clean_mask, create_mask_overlay
from flower_cbir.settings import PROJECT_ROOT, load_config


class DeepLabSegmenter:
    def __init__(self, config):
        self.checkpoint = (PROJECT_ROOT / config['checkpoint']).resolve()
        executable = config.get('python_executable') or sys.executable
        self.executable = str((PROJECT_ROOT / executable).resolve())
        if not self.checkpoint.is_file() or not Path(self.executable).is_file():
            raise FileNotFoundError('Missing DeepLabV3 checkpoint or Python runtime. See README.md.')
        self.lock = threading.RLock()
        self.responses = queue.Queue()
        log = PROJECT_ROOT / 'output/logs/deeplab_worker.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log_stream = log.open('a', encoding='utf-8')
        worker = (Path(config['worker'])
                  if config['method'] == 'deeplabv3plus_v2'
                  else Path(__file__).with_name('deeplab_worker.py'))
        if not worker.is_file():
            raise FileNotFoundError(f'Missing segmentation worker: {worker}')
        command = [self.executable, '-u', str(worker),
                   '--checkpoint', str(self.checkpoint), '--device', config['device']]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log_stream, text=True, encoding='utf-8', bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        def read_responses():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)
        threading.Thread(target=read_responses, daemon=True).start()
        atexit.register(self.close)
        self.info = self._read()
        if not self.info.get('ready'):
            self.close()
            raise RuntimeError('DeepLabV3 failed to initialize. See output/logs/deeplab_worker.log.')
        if self.info['checkpoint_sha256'] != config['_checkpoint_sha256']:
            self.close()
            raise RuntimeError('DeepLabV3 checkpoint changed while loading.')
        if config['method'] == 'deeplabv3plus_v2':
            from flower_cbir.core.contract import file_sha256
            if (self.info.get('model_config_sha256') != file_sha256(config['active_model_config'])
                    or self.info.get('postprocessing_sha256') != file_sha256(config['postprocessing_path'])
                    or self.info.get('postprocessing') != config['postprocessing']):
                self.close()
                raise RuntimeError('Selected model or postprocessing differs from CBIR configuration.')

    def _read(self):
        try:
            line = self.responses.get(timeout=90)
        except queue.Empty:
            self.close()
            raise RuntimeError('DeepLabV3 timed out. See output/logs/deeplab_worker.log.') from None
        if line is None:
            raise RuntimeError('DeepLabV3 worker stopped. See output/logs/deeplab_worker.log.')
        response = json.loads(line)
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response

    def predict(self, image_rgb):
        if image_rgb.shape != (384, 384, 3) or image_rgb.dtype != np.uint8:
            raise ValueError('DeepLabV3 requires uint8 RGB 384x384.')
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError('DeepLabV3 worker is no longer running.')
            self.process.stdin.write(json.dumps({'rgb': base64.b64encode(image_rgb.tobytes()).decode('ascii')}) + '\n')
            self.process.stdin.flush()
            response = self._read()
        mask = np.frombuffer(base64.b64decode(response['mask']), dtype=np.uint8).reshape(384, 384)
        return clean_mask(mask)

    def close(self):
        if hasattr(self, 'process') and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if hasattr(self, 'log_stream'):
            self.log_stream.close()


@lru_cache(maxsize=2)
def _get_segmenter(configuration):
    return DeepLabSegmenter(json.loads(configuration))


def get_segmenter():
    from flower_cbir.core.contract import file_sha256
    config = load_config()['segmentation']
    if config['method'] not in ('deeplabv3', 'deeplabv3plus_v2'):
        raise ValueError('Unsupported segmentation method; no silent fallback is allowed.')
    config['_checkpoint_sha256'] = file_sha256(PROJECT_ROOT / config['checkpoint'])
    key=json.dumps(config,sort_keys=True)
    segmenter=_get_segmenter(key)
    if segmenter.process.poll() is not None:
        _get_segmenter.cache_clear()
        segmenter=_get_segmenter(key)
    return segmenter


def run_deeplab(image_rgb):
    segmenter = get_segmenter()
    mask = segmenter.predict(image_rgb)
    return mask, create_mask_overlay(image_rgb, mask), segmenter.info
