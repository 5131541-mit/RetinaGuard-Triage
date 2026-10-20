"""Generate SYNTHETIC fundus-like images + labels for a pipeline smoke test (no downloads needed).
Metrics on this data are meaningless: it only proves train -> evaluate -> serve runs end to end.
python make_demo_data.py --n 200 --out data/demo"""
import argparse, os
import cv2, numpy as np, pandas as pd

def fundus(grade, seed, W=256):
    r = np.random.RandomState(seed); img = np.zeros((W, W, 3), np.uint8); c = W // 2; R = int(W * .47)
    yy, xx = np.mgrid[:W, :W]; d = np.hypot(xx - c, yy - c) / R
    for ch, (a, b) in enumerate([(25, 70), (95, 150), (190, 235)]): img[..., ch] = np.clip(b - (b - a) * d ** 2, 0, 255)
    dx = c + int(R * .5); cv2.circle(img, (dx, c), int(R * .13), (170, 220, 245), -1)
    for _ in range(14):
        a = r.uniform(0, 6.28); e = (int(c + np.cos(a) * R * .9), int(c + np.sin(a) * R * .9))
        cv2.line(img, (dx, c), e, (20, 25, 120), 2, cv2.LINE_AA)
    for _ in range(grade * 6 + r.randint(0, 3)):                   # dark red lesions
        a, q = r.uniform(0, 6.28), r.uniform(0, .7); cv2.circle(img, (int(c + np.cos(a) * R * q), int(c + np.sin(a) * R * q)), r.randint(1, 3 + grade), (15, 20, 110), -1)
    for _ in range(max(0, grade - 1) * 5):                          # bright exudates
        a, q = r.uniform(0, 6.28), r.uniform(0, .5); cv2.circle(img, (int(c + np.cos(a) * R * q), int(c + np.sin(a) * R * q)), r.randint(2, 4), (110, 230, 245), -1)
    img[d > 1] = 0
    return cv2.GaussianBlur(img, (3, 3), 0)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=200); ap.add_argument('--out', default='data/demo'); a = ap.parse_args()
    os.makedirs(f'{a.out}/images', exist_ok=True); r = np.random.RandomState(0); rows = []
    for i in range(a.n):
        g = int(r.choice(5, p=[.45, .1, .27, .08, .1])); cv2.imwrite(f'{a.out}/images/demo_{i:04d}.png', fundus(g, i)); rows.append((f'demo_{i:04d}', g))
    pd.DataFrame(rows, columns=['id_code', 'diagnosis']).to_csv(f'{a.out}/train.csv', index=False); print('wrote', a.n, 'images to', a.out)
