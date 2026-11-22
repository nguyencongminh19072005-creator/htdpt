"""Prepare reference masks and the fixed CBIR split for a new training run."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flower_cbir.core.contract import file_sha256
from flower_cbir.core.segmentation import clean_mask
from flower_cbir.evaluation.experiment import split_dataset
from flower_cbir.services.index_dataset import atomic_json
from flower_cbir.settings import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir',type=Path)
    parser.add_argument('--mask-dir',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'output/segmentation_data/split.json')
    args=parser.parse_args()
    config=load_config()['dataset']
    image_dir=args.image_dir or ROOT/config['image_dir']
    mask_dir=args.mask_dir or ROOT/config['reference_mask_dir']
    lookup=loadmat(ROOT/config['labels_mat'])['labels'].ravel()
    rows=[];destination=args.output.resolve().parent/'reference_masks'
    destination.mkdir(parents=True,exist_ok=True)
    for path in sorted(image_dir.glob('*.png')):
        source_id=int(path.stem.split('_')[-1])
        source_mask=mask_dir/path.name
        mask=clean_mask(np.asarray(Image.open(source_mask).convert('L')))
        with Image.open(path) as image:
            if image.mode!='RGB' or image.size!=(384,384) or mask.shape!=(384,384):
                raise ValueError(f'{path.name}: expected paired RGB image and mask at 384x384.')
        target=destination/path.name;Image.fromarray(mask).save(target)
        rows.append({'image_id':source_id,'source_id':source_id,'filename':path.name,
                     'image_path':str(path.resolve()),'mask_path':str(target),
                     'image_sha256':file_sha256(path),'mask_sha256':file_sha256(target),
                     'mask_source_sha256':file_sha256(source_mask),
                     'label':int(lookup[source_id-1]) if 1<=source_id<=len(lookup) else None})
    if len(rows)!=config['expected_count']:
        raise ValueError('Reference image count does not match the configured dataset.')
    splits=split_dataset(rows)
    output={name:[rows[int(i)] for i in indices]
            for name,indices in zip(['gallery','validation','test'],splits,strict=True)}
    output['provenance']='Supplied masks cleaned by the report rule; fixed seed 42; original-photo groups unverified.'
    atomic_json(args.output,output)
    print(json.dumps({'output':str(args.output),**{k:len(output[k]) for k in ['gallery','validation','test']}}))


if __name__=='__main__':main()
