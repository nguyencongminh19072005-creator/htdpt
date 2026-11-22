"""Lossless CSV interchange around the shared report-defined extraction pipeline."""
import csv
import json
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from flower_cbir.core.contract import file_sha256, verify_contract
from flower_cbir.core.deeplab import get_segmenter
from flower_cbir.core.normalization import SelectiveNormalizer
from flower_cbir.core.pipeline import extract_image
from flower_cbir.core.preprocess import load_image
from flower_cbir.core.schema import FEATURE_NAMES, validate_features
from flower_cbir.core.segmentation import clean_mask
from flower_cbir.services.index_dataset import atomic_json
from flower_cbir.settings import PROJECT_ROOT

META_NAMES = ['filename','image_sha256','mask_sha256','flower_pixels']


def export_csv(data, csv_path, manifest_path):
    csv_path, manifest_path = Path(csv_path),Path(manifest_path)
    verify_contract(data['contract'])
    validate_features(data['raw_features'])
    csv_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=csv_path.with_name(csv_path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(META_NAMES+FEATURE_NAMES)
        for row,vector in zip(data['metadata'],data['raw_features'],strict=True):
            mask=clean_mask(np.asarray(Image.open(row['mask_path']).convert('L')))
            writer.writerow([row['filename'],row['image_sha256'],row['mask_sha256'],int((mask>0).sum()),
                             *(format(float(x),'.17g') for x in vector)])
    os.replace(temporary,csv_path)
    records=[{**record,'image_path':Path(os.path.relpath(record['image_path'],PROJECT_ROOT)).as_posix(),
              'mask_path':Path(os.path.relpath(record['mask_path'],PROJECT_ROOT)).as_posix()}
             for record in data['metadata']]
    manifest={'format':'report1-cbir-csv-v3','csv_sha256':file_sha256(csv_path),
              'contract':data['contract'],'index_manifest':data['manifest'],
              'records':records,'normalizer':data['normalizer'].to_dict(),
              'summary':{'rows':len(data['metadata']),'dimensions':163,'groups':[78,18,28,39]}}
    atomic_json(manifest_path,manifest)
    return manifest['summary']


def validate_csv(path,expected_count=1000):
    rows=[];vectors=[];names=set()
    with Path(path).open('r',newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=META_NAMES+FEATURE_NAMES:
            raise ValueError('CSV header/order does not match the shared 163D schema.')
        for row in reader:
            if row['filename'] in names:
                raise ValueError('Duplicate filename in CSV.')
            names.add(row['filename'])
            values=np.array([float(row[k]) for k in FEATURE_NAMES],dtype=np.float64)
            vector=values.astype(np.float32)
            if not np.array_equal(values,vector.astype(np.float64)):
                raise ValueError('CSV is not a lossless float32 export of the current extractor; re-export it.')
            validate_features(vector)
            area=int(row['flower_pixels'])
            if not 0<area<=384*384:
                raise ValueError('Invalid flower area in CSV.')
            rows.append(row);vectors.append(vector)
    if not rows or (expected_count and len(rows)!=expected_count):
        raise ValueError(f'Expected {expected_count} CSV rows, found {len(rows)}.')
    return rows,np.array(vectors,dtype=np.float32)


def read_csv_batch(csv_path,manifest_path,expected_count=1000,image_dir=None,mask_dir=None,
                   verify_extraction=True):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('format')!='report1-cbir-csv-v3':
        raise ValueError('Legacy/unversioned CSV cannot be imported. Re-extract it with the shared pipeline.')
    verify_contract(manifest['contract'])
    if file_sha256(csv_path)!=manifest['csv_sha256']:
        raise ValueError('CSV checksum differs from the extraction manifest.')
    rows,raw=validate_csv(csv_path,expected_count)
    records=manifest['records']
    if [r['filename'] for r in rows]!=[r['filename'] for r in records]:
        raise ValueError('CSV and manifest image order differ.')
    if get_segmenter().info!=manifest['index_manifest']['model_runtime']:
        raise ValueError('DeepLabV3 runtime differs from the CSV extraction runtime.')
    metadata=[]
    for index,(row,original) in enumerate(zip(rows,records,strict=True)):
        record=dict(original)
        for kind,directory in [('image',image_dir),('mask',mask_dir)]:
            if directory is not None:
                record[kind+'_path']=str((Path(directory)/(Path(row['filename']).stem+'.png'
                    if kind=='mask' else row['filename'])).resolve())
            else:
                record[kind+'_path']=str((PROJECT_ROOT/record[kind+'_path']).resolve())
            if row[kind+'_sha256']!=record[kind+'_sha256'] or file_sha256(record[kind+'_path'])!=row[kind+'_sha256']:
                raise ValueError(f'Source {kind} changed since CSV extraction: {row["filename"]}')
        rgb=load_image(record['image_path'])
        mask=clean_mask(np.asarray(Image.open(record['mask_path']).convert('L')))
        if rgb.shape!=(384,384,3) or mask.shape!=(384,384) or int((mask>0).sum())!=int(row['flower_pixels']):
            raise ValueError(f'CSV source shape or flower area differs: {row["filename"]}')
        if verify_extraction:
            extracted=extract_image(rgb)
            if not np.array_equal(extracted['mask_binary'],mask):
                raise ValueError(f'{row["filename"]}: saved mask is not this DeepLabV3 prediction.')
            if not np.array_equal(extracted['raw_vector'],raw[index]):
                raise ValueError(f'{row["filename"]}: CSV vector is not bit-for-bit equal to the query extractor.')
        metadata.append(record)
        if (index+1)%100==0 or index+1==len(records):
            print(f'CSV/query parity verified: {index+1}/{len(records)}',flush=True)
    normalizer=SelectiveNormalizer().fit(raw)
    saved=SelectiveNormalizer.from_dict(manifest['normalizer'])
    if not np.array_equal(saved.mean,normalizer.mean) or not np.array_equal(saved.std,normalizer.std):
        raise ValueError('CSV normalizer was not fitted on its exported gallery.')
    return {'metadata':metadata,'raw_features':raw,'normalized_features':normalizer.transform(raw),
            'normalizer':normalizer,'version_id':manifest['index_manifest']['version_id'],
            'manifest':manifest['index_manifest'],'contract':manifest['contract']}
