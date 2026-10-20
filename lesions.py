"""OPTIONAL lesion localisation: classical blob heuristic (NOT a trained detector, NOT validated).
Types: 0=microaneurysm-like, 1=haemorrhage-like (dark blobs), 2=exudate-like (bright blobs).
Swap for a U-Net trained on IDRiD segmentation masks for real localisation."""
import cv2, numpy as np

def find_lesions(img, max_n=60):
    h, w = img.shape[:2]
    g = cv2.createCLAHE(3.0, (8, 8)).apply(img[:, :, 1]); bg = cv2.GaussianBlur(g, (0, 0), 20)
    yy, xx = np.mgrid[:h, :w]
    inside = np.hypot(xx - w / 2, yy - h / 2) < .46 * w
    s = cv2.GaussianBlur(g, (0, 0), 18); s[~inside] = 0
    dx, dy = cv2.minMaxLoc(s)[3]; disc = np.hypot(xx - dx, yy - dy) < .12 * w  # exclude optic disc
    ok = inside & ~disc
    dark = ((bg.astype(int) - g) > 22) & ok
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bright = (((g.astype(int) - bg) > 40) & (g > 200) & ok).astype(np.uint8)
    out = []
    for kind, mk in ((0, dark), (2, bright)):
        n, _, st, cen = cv2.connectedComponentsWithStats(mk)
        for i in range(1, n):
            a, bw, bh = st[i, 4], st[i, 2], st[i, 3]; fill = a / (bw * bh)
            if kind == 2 and 12 <= a <= 900: t = 2
            elif kind == 0 and 4 <= a <= 45 and max(bw, bh) <= 10: t = 0
            elif kind == 0 and 45 < a <= 900 and fill > .45 and max(bw, bh) < 3 * min(bw, bh): t = 1
            else: continue
            out.append((t, float(cen[i][0]), float(cen[i][1]), a))
    out.sort(key=lambda r: -r[3])
    return [(t, x, y) for t, x, y, _ in out[:max_n]]
