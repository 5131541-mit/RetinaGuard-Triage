"""Run: python tests/test_core.py   (torch-dependent tests are skipped if torch is missing)"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
from make_demo_data import fundus
from quality import quality, preprocess, crop_box
from lesions import find_lesions

def test_quality():
    good = quality(fundus(2, 1, 512)); bad = quality(np.zeros((512, 512, 3), np.uint8)); blur = quality(cv2.GaussianBlur(fundus(2, 1, 512), (0, 0), 12))
    assert min(good.values()) > 40 and sum(bad.values()) == 0 and blur['s'] < good['s'], (good, bad, blur)
def test_crop_preprocess():
    img = np.zeros((300, 500, 3), np.uint8); img[50:250, 150:350] = 120
    assert crop_box(img) == (150, 50, 200, 200) and preprocess(img, 128).shape == (128, 128, 3)
def test_lesions_scale_with_grade():
    n = lambda g: len(find_lesions(preprocess(fundus(g, 5, 512), 512)))
    assert n(4) > n(0), (n(0), n(4))
    assert all(t in (0, 1, 2) for t, _, _ in find_lesions(preprocess(fundus(3, 2, 512), 512)))
def test_ordinal_probs():
    try: import torch
    except ImportError: print('skip torch tests'); return
    from model import class_probs
    p = class_probs(torch.randn(8, 4)); assert torch.allclose(p.sum(1), torch.ones(8), atol=1e-5) and (p >= 0).all()
if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'): f(); print('PASS', k)
