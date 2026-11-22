"""Real end-to-end parity, not just d(v,v) on already stored vectors."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from flower_cbir.core.contract import fingerprint,file_sha256
from flower_cbir.core.schema import GROUP_SLICES
from flower_cbir.services.csv_tools import read_csv_batch
from flower_cbir.services.database import FlowerDatabase
from flower_cbir.services.index_dataset import atomic_json
from flower_cbir.services.search import FlowerSearchEngine
from flower_cbir.settings import PROJECT_ROOT


def verify_synchronization():
    from trich_rut_minh.full_163 import extract_one
    started=time.perf_counter()
    database=FlowerDatabase()
    try:
        data=database.load_index()
    finally:database.close()
    csv_data=read_csv_batch(PROJECT_ROOT/'trich_rut_minh/features_163.csv',
        PROJECT_ROOT/'trich_rut_minh/extraction_manifest_163.json',len(data['metadata']),verify_extraction=False)
    if csv_data['version_id']!=data['version_id']:
        raise ValueError('CSV and PostgreSQL refer to different index versions.')
    csv_rows={m['filename']:i for i,m in enumerate(csv_data['metadata'])}
    engine=FlowerSearchEngine(data)
    result={'version_id':engine.version_id,'contract_fingerprint':fingerprint(data['contract']),
            'count':len(data['metadata']),'raw_exact':0,'normalized_exact':0,'mask_exact':0,
            'csv_exact':0,'exporter_exact':0,'self_top1':0,'max_self_distance':0.,'failures':[]}
    result['csv_sha256']=file_sha256(PROJECT_ROOT/'trich_rut_minh/features_163.csv')
    result['verifier_sha256']=file_sha256(__file__)
    result['verified_through']=0
    progress=PROJECT_ROOT/'output/verification/synchronization_progress.json'
    if progress.is_file():
        previous=json.loads(progress.read_text(encoding='utf-8'))
        keys=['version_id','contract_fingerprint','csv_sha256','verifier_sha256','count']
        if all(previous.get(key)==result[key] for key in keys) and not previous.get('failures'):
            result=previous
            print(f'Resuming verified rows: {result["verified_through"]}/{engine.num_images}',flush=True)
    for index,item in enumerate(data['metadata']):
        if index<result['verified_through']:
            continue
        try:
            query=engine.search(item['image_path'],top_k=min(5,engine.num_images),
                                use_gallery_mask=False)
            info=query['query_info']
            saved_mask=np.asarray(Image.open(item['mask_path']).convert('L'))
            exporter=np.concatenate(extract_one(info['image_rgb'],info['mask_binary'])).astype(np.float32)
            checks={
                'raw_exact':np.array_equal(info['raw_vector'],data['raw_features'][index]),
                'normalized_exact':np.array_equal(info['norm_vector'],data['normalized_features'][index]),
                'mask_exact':np.array_equal(info['mask_binary'],saved_mask),
                'csv_exact':np.array_equal(info['raw_vector'],csv_data['raw_features'][csv_rows[item['filename']]]),
                'exporter_exact':np.array_equal(info['raw_vector'],exporter),
            }
            for name,ok in checks.items():
                result[name]+=int(ok)
                if not ok:result['failures'].append({'filename':item['filename'],'check':name})
            self_hit=next((hit for hit in query['results'] if hit['filename']==item['filename']),None)
            if self_hit is not None:
                result['max_self_distance']=max(result['max_self_distance'],self_hit['distance'])
                result['self_top1']+=int(self_hit['rank']==1)
            else:result['failures'].append({'filename':item['filename'],'check':'self_not_in_top5'})
        except Exception as exc:
            result['failures'].append({'filename':item['filename'],'error':str(exc)})
        if (index+1)%100==0 or index+1==engine.num_images:
            result['verified_through']=index+1
            atomic_json(progress,result)
            print(f'Fresh DeepLabV3+ postprocessing/query/CSV/PostgreSQL parity: {index+1}/{engine.num_images}; failures={len(result["failures"])}',flush=True)
    result['seconds']=time.perf_counter()-started
    result['passed']=not result['failures'] and result['max_self_distance']==0 and all(
        result[key]==engine.num_images for key in ['raw_exact','normalized_exact','mask_exact','csv_exact','exporter_exact'])
    atomic_json(PROJECT_ROOT/'output/verification/synchronization.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:
        raise ValueError('End-to-end synchronization verification failed; see output/verification/synchronization.json.')
    return result
