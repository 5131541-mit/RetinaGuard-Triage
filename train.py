"""Train + evaluate. Example:
python train.py --train "aptos|data/aptos/train.csv|data/aptos/train_images" \
                --ext "idrid|data/idrid/labels.csv|data/idrid/images" --epochs 15
Dataset spec = name|labels.csv|image_dir  (APTOS: id_code,diagnosis | IDRiD: 'Image name','Retinopathy grade' | EyePACS: image,level)"""
import argparse, json, os, copy, hashlib
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, cv2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from model import RetinaNet, class_probs, K, GRADES, MEAN, STD
from quality import quality, preprocess

IDC, LBC = ['id_code', 'Image name', 'image', 'image_id'], ['diagnosis', 'Retinopathy grade', 'level', 'label']
PATIENT_COLS = ['patient_id', 'patient', 'patient_id_code', 'subject_id']

def read(spec, require_stratify=False):
    parts = spec.split('|')
    if len(parts) != 3 or not all(parts): raise ValueError('Dataset spec must be name|labels.csv|image_dir')
    name, csv, d = parts
    if not os.path.isfile(csv): raise FileNotFoundError(f'Labels CSV not found: {csv}')
    if not os.path.isdir(d): raise FileNotFoundError(f'Image directory not found: {d}')
    df = pd.read_csv(csv); df.columns = [str(c).strip() for c in df.columns]
    ic = next((c for c in IDC if c in df.columns), None); lc = next((c for c in LBC if c in df.columns), None)
    if ic is None or lc is None: raise ValueError(f'{csv}: expected an image ID column ({IDC}) and label column ({LBC}); found {list(df.columns)}')
    labels = pd.to_numeric(df[lc], errors='coerce')
    invalid = labels.isna() | (labels % 1 != 0) | (labels < 0) | (labels >= K)
    if invalid.any(): raise ValueError(f'{csv}: {int(invalid.sum())} invalid labels; grades must be integers 0-4')
    if df[ic].astype(str).duplicated().any(): raise ValueError(f'{csv}: duplicate image IDs detected; remove duplicates before training')
    def path(i):
        for e in ('', '.png', '.jpg', '.jpeg', '.JPG', '.PNG', '.JPEG'):
            candidate = os.path.realpath(os.path.join(d, str(i) + e))
            try: within = os.path.commonpath([os.path.realpath(d), candidate]) == os.path.realpath(d)
            except ValueError: within = False
            if within and os.path.isfile(candidate): return candidate
        return None
    paths = [path(i) for i in df[ic]]
    missing = sum(p is None for p in paths)
    if missing: raise ValueError(f'{csv}: {missing}/{len(paths)} labelled images were not found in {d}; refusing silent row drops')
    # Verify actual image decoding and detect exact byte duplicates across distinct IDs.
    hashes = []; phashes = []
    for image_path in paths:
        try:
            with open(image_path, 'rb') as fh: digest = hashlib.sha256(fh.read()).hexdigest()
        except OSError as exc:
            raise ValueError(f'Cannot read image {image_path}: {exc}') from exc
        decoded = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f'Corrupt or unsupported image file: {image_path}')
        hashes.append(digest)
        gray = cv2.cvtColor(cv2.resize(decoded, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        dct = cv2.dct(gray)[:8, :8].reshape(-1)
        med = np.median(dct[1:])
        phashes.append(sum((int(v > med) << i) for i, v in enumerate(dct)))
    hash_series = pd.Series(hashes)
    duplicate_hash = hash_series.duplicated(keep=False)
    if duplicate_hash.any():
        examples = [str(df[ic].iloc[i]) for i in np.flatnonzero(duplicate_hash.to_numpy())[:8]]
        raise ValueError(f'{csv}: exact duplicate image content detected under multiple rows/IDs; examples={examples}. Deduplicate before splitting.')
    # Perceptual-hash screening catches obvious resized/recompressed near-duplicates.
    # Bucket by high bits to keep comparisons tractable; this is a screening heuristic, not proof of identity.
    buckets = {}
    for i, ph in enumerate(phashes):
        # Four 16-bit bands: any pair with Hamming distance <=3 shares at least one band.
        for band in range(4):
            key = (band, (ph >> (band * 16)) & 0xffff)
            buckets.setdefault(key, []).append((i, ph))
    near_pairs = []; seen_pairs = set()
    for members in buckets.values():
        for pos, (i, hi) in enumerate(members):
            for j, hj in members[pos + 1:]:
                pair = (min(i, j), max(i, j))
                if pair in seen_pairs: continue
                seen_pairs.add(pair)
                if (hi ^ hj).bit_count() <= 3:
                    near_pairs.append(pair)
                    if len(near_pairs) >= 20: break
            if len(near_pairs) >= 20: break
        if len(near_pairs) >= 20: break
    if near_pairs:
        examples = [(str(df[ic].iloc[i]), str(df[ic].iloc[j])) for i, j in near_pairs[:5]]
        raise ValueError(f'{csv}: possible near-duplicate images detected by perceptual hash; review these pairs before training: {examples}')
    patient_col = next((c for c in PATIENT_COLS if c in df.columns), None)
    data = {'id': df[ic].astype(str), 'y': labels.astype(int), 'path': paths, 'ds': name, 'sha256': hashes}
    if patient_col:
        if df[patient_col].isna().any(): raise ValueError(f'{csv}: missing values in patient identifier column {patient_col}')
        data['patient_id'] = df[patient_col].astype(str)
    o = pd.DataFrame(data)
    if len(o) == 0: raise ValueError(f'{csv}: no usable images found')
    if require_stratify:
        if len(o) < 50: raise ValueError(f'{csv}: only {len(o)} images; at least 50 are required for stratified train/validation/test splits')
        counts = o.y.value_counts()
        rare = [int(k) for k in range(K) if counts.get(k, 0) < 10]
        if rare: raise ValueError(f'{csv}: each grade needs at least 10 images for stratified splitting; insufficient grades: {rare}')
    return o.reset_index(drop=True)

class DS(Dataset):
    def __init__(s, df, size, aug):
        s.df = df.reset_index(drop=True)
        base = [T.ToTensor(), T.Normalize(MEAN, STD)]
        s.tf = T.Compose([T.ToPILImage(), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(180),
                          T.ColorJitter(.2, .2, .1), T.RandomResizedCrop(size, (.85, 1))] + base) if aug else T.Compose(base)
        s.size = size
    def __len__(s): return len(s.df)
    def __getitem__(s, i):
        r = s.df.iloc[i]; raw = cv2.imread(r.path)
        q = min(quality(raw).values())
        return s.tf(cv2.cvtColor(preprocess(raw, s.size), cv2.COLOR_BGR2RGB)), int(r.y), float(q)

def cum(y): return (y[:, None] > torch.arange(K - 1)).float()

@torch.no_grad()
def run(m, dl, dev):
    m.eval(); L, Y, Q = [], [], []
    for x, y, q in dl: L.append(m(x.to(dev)).float().cpu()); Y.append(y); Q.append(q)
    return torch.cat(L), torch.cat(Y).numpy(), torch.cat(Q).numpy()

def metrics(L, Y):
    c = torch.cummin(torch.sigmoid(L), 1).values.numpy(); pr = (c > .5).sum(1); ref = c[:, 1]   # ref = P(grade>=2)
    conf = np.where(ref > .5, ref, 1 - ref); ok = ((ref > .5) == (Y >= 2)).astype(float); ece = 0.
    for lo in np.linspace(0, .9, 10):
        m = (conf > lo) & (conf <= lo + .1)
        if m.any(): ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
    cm = confusion_matrix(Y, pr, labels=range(K))
    per_grade = {}
    total = int(cm.sum())
    for grade in range(K):
        tp = int(cm[grade, grade]); fn = int(cm[grade, :].sum() - tp)
        fp = int(cm[:, grade].sum() - tp); tn = total - tp - fn - fp
        per_grade[GRADES[grade]] = {
            'sensitivity': float(tp / (tp + fn)) if tp + fn else None,
            'specificity': float(tn / (tn + fp)) if tn + fp else None,
            'support': int(tp + fn),
        }
    return dict(n=int(len(Y)), acc=float((pr == Y).mean()), qwk=float(cohen_kappa_score(Y, pr, weights='quadratic')),
                macro_f1=float(f1_score(Y, pr, average='macro')),
                auroc=float(roc_auc_score(Y >= 2, ref)) if len(set(Y >= 2)) > 1 else None, ece=float(ece),
                ece_scope='binary referable DR (grade >= 2), not multiclass calibration', per_grade=per_grade), pr

def fit_T(L, Y):
    t = torch.ones(1, requires_grad=True); tg = cum(torch.tensor(Y)); o = torch.optim.LBFGS([t], lr=.1, max_iter=100)
    def f():
        o.zero_grad(); l = F.binary_cross_entropy_with_logits(L / t, tg); l.backward(); return l
    o.step(f); return t.detach().clamp(.5, 5)

def error_analysis(te, pr, L, Q, out):
    d = te.copy(); P = class_probs(L).numpy()
    d['pred'], d['conf'], d['quality'] = pr, P[np.arange(len(pr)), pr], Q; d['abs_err'] = abs(d.pred - d.y)
    d[d.abs_err > 0].sort_values('abs_err', ascending=False).to_csv(f'{out}/errors.csv', index=False)
    cm = confusion_matrix(d.y, d.pred, labels=range(K)); plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap='Blues')
    for i in range(K):
        for j in range(K): plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.xticks(range(K), range(K)); plt.yticks(range(K), range(K)); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion matrix (test)'); plt.tight_layout(); plt.savefig(f'{out}/confusion_matrix.png', dpi=130); plt.close()
    bins = pd.cut(d.quality, [-1, 50, 70, 101], labels=['<50', '50-70', '>=70'])
    return dict(error_rate_by_class={GRADES[k]: float((g.abs_err > 0).mean()) for k, g in d.groupby('y')},
                error_rate_by_quality={str(k): float((g.abs_err > 0).mean()) for k, g in d.groupby(bins, observed=True)},
                off_by_2_or_more=int((d.abs_err >= 2).sum()), mean_conf_when_wrong=float(d[d.abs_err > 0].conf.mean()),
                mean_conf_when_right=float(d[d.abs_err == 0].conf.mean()),
                per_class=classification_report(d.y, d.pred, labels=range(K), target_names=GRADES, output_dict=True, zero_division=0))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--ext', nargs='*', default=[])
    ap.add_argument('--epochs', type=int, default=15); ap.add_argument('--bs', type=int, default=16); ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--out', default='outputs'); ap.add_argument('--workers', type=int, default=2); ap.add_argument('--no-pretrained', action='store_true', help='Do not download/use ImageNet weights (useful for offline smoke tests only)')
    a = ap.parse_args();
    if not 1 <= a.epochs <= 100: ap.error('--epochs must be 1..100')
    if not 1 <= a.bs <= 256: ap.error('--bs must be 1..256')
    if a.size not in (256, 384, 512, 768, 1024): ap.error('--size must be 256, 384, 512, 768 or 1024')
    os.makedirs(a.out, exist_ok=True); torch.manual_seed(0); np.random.seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = read(a.train, require_stratify=True)
    if 'patient_id' in df.columns:
        # Group-based splits prevent patient overlap. If class coverage is inadequate, stop rather than leak.
        first = GroupShuffleSplit(n_splits=1, test_size=.30, random_state=0)
        tr_idx, hold_idx = next(first.split(df, df.y, groups=df.patient_id))
        tr, hold = df.iloc[tr_idx].copy(), df.iloc[hold_idx].copy()
        second = GroupShuffleSplit(n_splits=1, test_size=.50, random_state=1)
        va_rel, te_rel = next(second.split(hold, hold.y, groups=hold.patient_id))
        va, te = hold.iloc[va_rel].copy(), hold.iloc[te_rel].copy()
        for a_name, part in (('train', tr), ('val', va), ('test', te)):
            if set(part.y.unique()) != set(range(K)):
                raise ValueError(f'Patient-level {a_name} split lacks one or more grades. Add more patients/data; do not fall back to image-level splitting.')
        groups = [set(part.patient_id) for part in (tr, va, te)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise RuntimeError('Patient leakage detected between splits.')
        split_mode = 'patient-level'
    else:
        tr, te = train_test_split(df, test_size=.3, stratify=df.y, random_state=0)
        va, te = train_test_split(te, test_size=.5, stratify=te.y, random_state=0)
        split_mode = 'image-level; patient identifiers unavailable, residual patient leakage risk'
    for n, d in (('train', tr), ('val', va), ('test', te)): d.to_csv(f'{a.out}/split_{n}.csv', index=False)
    cnt = np.bincount(tr.y, minlength=K); print('train class counts', cnt)
    sampler = WeightedRandomSampler((1 / np.sqrt(cnt + 1))[tr.y.values], len(tr), True)   # class-imbalance strategy
    mk = lambda d, aug, **k: DataLoader(DS(d, a.size, aug), batch_size=a.bs, num_workers=a.workers, **k)
    dtr, dva, dte = mk(tr, True, sampler=sampler), mk(va, False), mk(te, False)
    m = RetinaNet(pretrained=not a.no_pretrained).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs); sc = torch.cuda.amp.GradScaler(enabled=dev == 'cuda')
    best, bs, hist = -1, None, []
    for ep in range(a.epochs):
        m.train(); tl = 0
        for x, y, _ in dtr:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=dev == 'cuda'): loss = F.binary_cross_entropy_with_logits(m(x).float(), cum(y.cpu()).to(dev))
            opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update(); tl += loss.item() * len(y)
        sch.step(); L, Y, _ = run(m, dva, dev); mv, _ = metrics(L, Y)
        hist.append(dict(epoch=ep + 1, train_loss=tl / len(tr), val_loss=float(F.binary_cross_entropy_with_logits(L, cum(torch.tensor(Y)))), val_qwk=mv['qwk']))
        print(hist[-1])
        if mv['qwk'] > best: best, bs = mv['qwk'], copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); L, Y, _ = run(m, dva, dev); m.T.data = fit_T(L, Y).to(dev)      # temperature scaling on val
    L, Y, Q = run(m, dte, dev); mt, pr = metrics(L, Y)
    res = dict(model='EfficientNet-B3 (ordinal)', split_mode=split_mode, dataset=str(a.train.split('|')[0]), **mt, T=float(m.T), train_class_counts=cnt.tolist(), history=hist,
               errors=error_analysis(te, pr, L, Q, a.out), cross_dataset={})
    for spec in a.ext:                                                                     # cross-dataset evaluation
        e = read(spec); Le, Ye, _ = run(m, mk(e, False), dev); res['cross_dataset'][e.ds[0]] = metrics(Le, Ye)[0]
    res['limitations'] = [f'Split mode: {split_mode}. If patient IDs are unavailable, patient-level leakage cannot be ruled out.', 'Class imbalance: severe/proliferative are rare.',
                          'No demographic labels: subgroup performance untested.', 'Camera/site shift: see cross_dataset.', 'Research prototype, not a medical device.']
    torch.save(dict(state=m.state_dict(), name=res['model'], size=a.size), f'{a.out}/best.pt')
    json.dump(res, open(f'{a.out}/metrics.json', 'w'), indent=1); print(json.dumps({k: res[k] for k in ('qwk', 'macro_f1', 'auroc', 'ece', 'cross_dataset')}, indent=1))

if __name__ == '__main__': main()
