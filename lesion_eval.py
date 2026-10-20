"""Annotation-based lesion-mask evaluation for research only.

Expected CSV columns: image, mask (relative to --root or absolute), optional lesion_type.
For multi-class masks, evaluate pixels > 0 as lesion; per-class reports require one CSV per class.
This evaluates the current heuristic candidate mask, NOT a trained U-Net.
"""
import argparse, csv, json, os
import cv2
import numpy as np
from lesions import find_lesions
from quality import preprocess

def score(pred, truth):
    p = pred.astype(bool); t = truth.astype(bool)
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'dice':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 1.0,
            'iou':float(tp/(tp+fp+fn)) if tp+fp+fn else 1.0,
            'precision':float(tp/(tp+fp)) if tp+fp else 1.0,
            'recall':float(tp/(tp+fn)) if tp+fn else 1.0}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--root',default='.'); ap.add_argument('--out',default='outputs/lesion_metrics.json'); a=ap.parse_args()
    rows=[]
    with open(a.csv,newline='',encoding='utf-8-sig') as f:
        rd=csv.DictReader(f)
        if not {'image','mask'}.issubset(rd.fieldnames or []): raise SystemExit('CSV must contain image,mask columns.')
        for r in rd:
            ip=r['image'] if os.path.isabs(r['image']) else os.path.join(a.root,r['image'])
            mp=r['mask'] if os.path.isabs(r['mask']) else os.path.join(a.root,r['mask'])
            im=cv2.imread(ip); mask=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
            if im is None or mask is None: raise SystemExit(f'Missing/unreadable image or mask: {ip} / {mp}')
            size=512; x=preprocess(im,size); truth=cv2.resize((mask>0).astype(np.uint8),(size,size),interpolation=cv2.INTER_NEAREST)
            cand=find_lesions(x); pred=np.zeros((size,size),np.uint8)
            for lesion in cand:
                px=int((lesion['x']+1)*.5*size); py=int((lesion['y']+1)*.5*size)
                if 0<=px<size and 0<=py<size: cv2.circle(pred,(px,py),3,1,-1)
            rows.append({'image':r['image'],**score(pred,truth)})
    if not rows: raise SystemExit('No annotation rows found.')
    keys=['dice','iou','precision','recall']; result={'n_images':len(rows),'method':'heuristic blob candidate circles vs supplied binary masks','metrics':{k:float(np.mean([x[k] for x in rows])) for k in keys},'per_image':rows,'limitations':['This is not a trained lesion segmentation model.','Pixel metrics depend on mask format and the circle rasterization tolerance.','Do not interpret as clinical validity.']}
    os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True); json.dump(result,open(a.out,'w'),indent=2); print(json.dumps(result['metrics'],indent=2))
if __name__=='__main__': main()
