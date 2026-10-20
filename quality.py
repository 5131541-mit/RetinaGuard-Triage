"""Image-quality gate + preprocessing (crop black border, resize, CLAHE). Thresholds are heuristics: calibrate on your data."""
import cv2, numpy as np

def crop_box(img, thr=15):
    ys, xs = np.where(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > thr)
    if len(xs) < 100: return 0, 0, img.shape[1], img.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)

def preprocess(img, size=512):
    x, y, w, h = crop_box(img)
    img = cv2.resize(img[y:y + h, x:x + w], (size, size))
    l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def quality(raw):
    """Return brightness/contrast/sharpness/field-coverage scores, each 0-100 (b,c,s,f)."""
    g = cv2.cvtColor(cv2.resize(raw, (512, 512)), cv2.COLOR_BGR2GRAY)
    m = g > 15
    if m.sum() < 1000: return dict(b=0, c=0, s=0, f=0)
    v = g[m].astype(float)
    lap = cv2.Laplacian(g, cv2.CV_64F)[m].var()
    cov = (cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) > 15).mean()
    cl = lambda z: int(max(0, min(100, z)))
    return dict(b=cl(100 - abs(v.mean() - 100) * .9), c=cl(v.std() * 2.2), s=cl(lap / 1.5), f=cl(cov / .7 * 100))
