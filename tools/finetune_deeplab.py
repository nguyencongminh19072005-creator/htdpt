"""Fine-tune segmentation heads on gallery masks; select only on validation Dice.

The split deliberately matches CBIR gallery/validation/test. Test images never
participate in optimization, checkpoint selection or early stopping.
Run with the configured PyTorch Python, not necessarily the UI's Python.
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import deeplabv3_resnet50


class Flowers(Dataset):
    def __init__(self, rows, augment=False):
        self.rows, self.augment = rows, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['image_path']) as source, Image.open(row['mask_path']) as target:
            image, mask = source.convert('RGB'), target.convert('L')
        if self.augment:
            for method in [Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM]:
                if random.random() < .5:
                    image, mask = image.transpose(method), mask.transpose(method)
            k = random.randrange(4)
            if k:
                angle = 90*k
                image, mask = image.rotate(angle), mask.rotate(angle)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(.85,1.15))
        array = np.asarray(image,dtype=np.float32).transpose(2,0,1)/255.0
        array = (array-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]
        labels = (np.asarray(mask)>0).astype(np.int64)
        return torch.from_numpy(array.copy()),torch.from_numpy(labels.copy())


def evaluate(model, loader, device):
    model.eval()
    tp=fp=fn=empty=0
    with torch.inference_mode():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            pred=model(x)['out'].argmax(1)
            tp+=int(((pred==1)&(y==1)).sum())
            fp+=int(((pred==1)&(y==0)).sum())
            fn+=int(((pred==0)&(y==1)).sum())
            empty+=int((pred.flatten(1).sum(1)==0).sum())
    return {'dice':2*tp/(2*tp+fp+fn),'iou':tp/(tp+fp+fn),'empty_masks':empty,'tp':tp,'fp':fp,'fn':fn}


def binary_cross_entropy_logits(logits, labels):
    # The CUDA NLLLoss2d kernel in this installed runtime has no deterministic
    # implementation. This is the same two-class cross entropy without that kernel.
    log_probability=logits.log_softmax(1)
    truth=labels.to(logits.dtype)
    return -((1-truth)*log_probability[:,0]+truth*log_probability[:,1]).mean()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--patience',type=int,default=3)
    args=parser.parse_args()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    train=DataLoader(Flowers(manifest['gallery'],True),batch_size=2,shuffle=True,drop_last=True,num_workers=0)
    validation=DataLoader(Flowers(manifest['validation']),batch_size=2,shuffle=False,num_workers=0)
    test=DataLoader(Flowers(manifest['test']),batch_size=2,shuffle=False,num_workers=0)
    model=deeplabv3_resnet50(weights=None,weights_backbone=None,num_classes=2,aux_loss=True)
    payload=torch.load(args.source,map_location='cpu',weights_only=True)
    model.load_state_dict(payload['model'],strict=True)
    for parameter in model.backbone.parameters():parameter.requires_grad_(False)
    model.to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=1)
    args.output.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256(args.source.read_bytes()).hexdigest()
    manifest_hash=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    baseline=evaluate(model,validation,device)
    best=baseline['dice'];stale=0;best_epoch=0
    history=[]
    def save_best(epoch,metrics):
        torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'architecture':'deeplabv3_resnet50_2classes_aux',
                    'training_source_sha256':source_hash,'training_split_sha256':manifest_hash,
                    'epoch':epoch,'best_validation_dice':metrics['dice']},args.output/'best.pt')
    save_best(0,baseline)
    print('BASELINE',json.dumps(baseline),flush=True)
    for epoch in range(1,args.epochs+1):
        started=time.perf_counter();model.train();model.backbone.eval()
        losses=[]
        for step,(x,y) in enumerate(train,1):
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs=model(x)
            logits=outputs['out']
            ce=binary_cross_entropy_logits(logits,y)
            p=logits.softmax(1)[:,1];truth=(y==1).float()
            dice=1-((2*(p*truth).sum((1,2))+1)/(p.sum((1,2))+truth.sum((1,2))+1)).mean()
            loss=ce+dice+.4*binary_cross_entropy_logits(outputs['aux'],y)
            loss.backward();optimizer.step()
            losses.append(loss.item())
            if step%100==0:print(f'epoch {epoch} step {step}/{len(train)} loss {np.mean(losses):.5f}',flush=True)
        metrics=evaluate(model,validation,device)
        scheduler.step(metrics['dice'])
        record={'epoch':epoch,'loss':float(np.mean(losses)),'validation':metrics,'seconds':time.perf_counter()-started}
        history.append(record)
        if metrics['dice']>best:
            best=metrics['dice'];best_epoch=epoch;stale=0;save_best(epoch,metrics)
        else:stale+=1
        (args.output/'history.json').write_text(json.dumps({'baseline_validation':baseline,'history':history,'best_epoch':best_epoch},indent=2),encoding='utf-8')
        print('EPOCH',json.dumps(record),flush=True)
        if stale>=args.patience:break
    model.load_state_dict(torch.load(args.output/'best.pt',map_location=device,weights_only=True)['model'])
    results={'baseline_validation':baseline,'best_epoch':best_epoch,'best_validation_dice':best,
             'test':evaluate(model,test,device),'split_counts':{k:len(manifest[k]) for k in ['gallery','validation','test']},
             'selection':'validation Dice only; frozen backbone; test evaluated once after selection',
             'training_source_sha256':source_hash,'split_sha256':manifest_hash,'history':history}
    (args.output/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('FINAL',json.dumps({k:v for k,v in results.items() if k!='history'}),flush=True)


if __name__=='__main__':main()
