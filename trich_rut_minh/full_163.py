"""CSV/PostgreSQL tools using EXACTLY the application's report-defined algorithms.

There are no independent feature formulas here. Run from any working directory:
    python trich_rut_minh/full_163.py all
    python trich_rut_minh/full_163.py check
    python trich_rut_minh/full_163.py load
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from flower_cbir.core.features import extract_all_features
from flower_cbir.core.preprocess import load_image
from flower_cbir.core.schema import (FEATURE_NAMES,F1_NAMES,F2_NAMES,F3_NAMES,F4_NAMES,
                                     GROUP_LENGTHS,GROUP_SLICES,EXTRACTOR_VERSION)
from flower_cbir.core.segmentation import clean_mask
from flower_cbir.services.csv_tools import export_csv,read_csv_batch,validate_csv as _validate_csv
from flower_cbir.services.database import FlowerDatabase,load_database_config
from flower_cbir.services.index_dataset import build_or_load_index

CSV=HERE/'features_163.csv'
MANIFEST=HERE/'extraction_manifest_163.json'
VERSION=EXTRACTOR_VERSION


def read_pair(image_path,mask_path):
    rgb=load_image(image_path)
    mask=clean_mask(np.asarray(Image.open(mask_path).convert('L')))
    if rgb.shape!=(384,384,3) or mask.shape!=(384,384):
        raise ValueError('Prepared image and mask must be 384x384.')
    return rgb,mask>0


def extract_one(rgb,mask):
    """Compatibility API: same function object and formulas as query extraction."""
    vector,_=extract_all_features(rgb,mask)
    return tuple(vector[part].tolist() for part in GROUP_SLICES)


def validate_csv(path,expected_count=1000):
    rows,_=_validate_csv(path,expected_count)
    return {'rows':len(rows),'dimensions':163,'groups':list(GROUP_LENGTHS)}


def extract(args):
    data=build_or_load_index(image_dir=args.image_dir,mask_dir=args.mask_dir,
                            expected_count=args.expected_count)
    print(json.dumps(export_csv(data,args.csv,args.manifest),ensure_ascii=False))


def load(args):
    data=read_csv_batch(args.csv,args.manifest,args.expected_count,args.image_dir,args.mask_dir)
    config=load_database_config()
    for key,attribute in [('host','host'),('port','port'),('dbname','database'),('user','user')]:
        value=getattr(args,attribute,None)
        if value is not None:config[key]=value
    database=FlowerDatabase(config)
    try:
        result=database.publish_index(data)
        loaded=database.load_index()
        by_name={m['filename']:i for i,m in enumerate(loaded['metadata'])}
        for index,item in enumerate(data['metadata']):
            other=by_name[item['filename']]
            if not np.array_equal(data['raw_features'][index],loaded['raw_features'][other]) or not np.array_equal(
                    data['normalized_features'][index],loaded['normalized_features'][other]):
                raise ValueError('PostgreSQL round-trip parity check failed.')
        print(json.dumps({**result,'postgres_round_trip':'bitwise_exact'},ensure_ascii=False))
    finally:database.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('extract','check','load','all'))
    parser.add_argument('--image-dir',type=Path,default=None)
    parser.add_argument('--mask-dir',type=Path,default=None,
                        help='Optional supplied DeepLabV3 masks; checked against the configured model.')
    parser.add_argument('--csv',type=Path,default=CSV)
    parser.add_argument('--manifest',type=Path,default=MANIFEST)
    parser.add_argument('--expected-count',type=int,default=1000)
    parser.add_argument('--host',default=None)
    parser.add_argument('--port',type=int,default=None)
    parser.add_argument('--database',default=None)
    parser.add_argument('--user',default=None)
    args=parser.parse_args()
    if args.action in ('extract','all'):extract(args)
    if args.action=='check':
        data=read_csv_batch(args.csv,args.manifest,args.expected_count,args.image_dir,args.mask_dir)
        print(json.dumps({'rows':len(data['metadata']),'dimensions':163,'query_parity':'bitwise_exact'}))
    if args.action in ('load','all'):load(args)


if __name__=='__main__':main()
