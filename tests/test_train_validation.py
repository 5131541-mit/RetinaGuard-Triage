import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np, pandas as pd, pytest
from train import read

def build(root, n=50, missing=False):
    imdir=os.path.join(root,'images'); os.makedirs(imdir)
    rows=[]
    for i in range(n):
        grade=i%5; rows.append({'id_code':f'im{i}','diagnosis':grade})
        if not (missing and i==0):
            rng=np.random.default_rng(i); img=np.clip(rng.normal(40+grade*30, 18, (32,32,3)),0,255).astype(np.uint8)
            cv2.imwrite(os.path.join(imdir,f'im{i}.png'),img)
    csv=os.path.join(root,'labels.csv'); pd.DataFrame(rows).to_csv(csv,index=False)
    return csv,imdir

def test_read_validates_dataset_and_preserves_labels():
    with tempfile.TemporaryDirectory() as d:
        csv,im=build(d); df=read(f'aptos|{csv}|{im}',require_stratify=True)
        assert len(df)==50 and set(df.y)==set(range(5))

def test_read_rejects_missing_image_instead_of_silently_dropping():
    with tempfile.TemporaryDirectory() as d:
        csv,im=build(d,missing=True)
        with pytest.raises(ValueError,match='not found'): read(f'aptos|{csv}|{im}',require_stratify=True)

def test_read_rejects_invalid_label():
    with tempfile.TemporaryDirectory() as d:
        csv,im=build(d); x=pd.read_csv(csv); x.loc[0,'diagnosis']=8; x.to_csv(csv,index=False)
        with pytest.raises(ValueError,match='invalid labels'): read(f'aptos|{csv}|{im}',require_stratify=True)


def test_read_rejects_exact_duplicate_image_content():
    with tempfile.TemporaryDirectory() as d:
        csv, im = build(d)
        import shutil
        shutil.copy(os.path.join(im, 'im0.png'), os.path.join(im, 'im50.png'))
        labels = pd.read_csv(csv)
        labels.loc[len(labels)] = {'id_code': 'im50', 'diagnosis': 0}
        labels.to_csv(csv, index=False)
        with pytest.raises(ValueError, match='exact duplicate image content'):
            read(f'aptos|{csv}|{im}', require_stratify=False)

def test_read_rejects_corrupt_image():
    with tempfile.TemporaryDirectory() as d:
        csv, im = build(d)
        with open(os.path.join(im, 'im0.png'), 'wb') as f: f.write(b'not an image')
        with pytest.raises(ValueError, match='Corrupt or unsupported image'):
            read(f'aptos|{csv}|{im}', require_stratify=False)
