"""Dataset acquisition helpers for RetinaGuard Triage.

Kaggle downloads require a Kaggle account, competition access/terms acceptance,
and local Kaggle API credentials. This script never bypasses access controls.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATASETS = {
    'aptos': {'competition': 'aptos2019-blindness-detection', 'csv': 'train.csv', 'images': 'train_images', 'id_col': 'id_code', 'label_col': 'diagnosis'},
    'eyepacs': {'competition': 'diabetic-retinopathy-detection', 'csv': 'trainLabels.csv', 'images': 'train', 'id_col': 'image', 'label_col': 'level'},
}


def download(name: str):
    cfg = DATASETS[name]
    dest = DATA / name
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which('kaggle') is None:
        raise RuntimeError('Kaggle CLI is not installed. Run: python -m pip install kaggle')
    try:
        subprocess.run(['kaggle', 'competitions', 'download', '-c', cfg['competition'], '-p', str(dest)], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'Kaggle download failed for {name}. Confirm Kaggle API credentials and accept competition rules at https://www.kaggle.com/competitions/{cfg["competition"]}/data') from e
    archives = list(dest.glob('*.zip'))
    if archives:
        import zipfile
        for archive in archives:
            print(f'Extracting {archive.name} ...')
            with zipfile.ZipFile(archive) as zf:
                # Extract only within the selected dataset directory.
                base = dest.resolve()
                for member in zf.infolist():
                    target = (dest / member.filename).resolve()
                    if target != base and base not in target.parents:
                        raise RuntimeError(f'Unsafe archive path: {member.filename}')
                zf.extractall(dest)
    csv_path = dest / cfg['csv']
    image_dir = dest / cfg['images']
    if not csv_path.exists():
        # EyePACS downloads can arrive with a slightly different archive layout.
        matches = list(dest.rglob(cfg['csv']))
        if matches:
            csv_path = matches[0]
    if not csv_path.exists():
        raise RuntimeError(f'Download extracted, but expected label file {cfg["csv"]} was not found under {dest}. Inspect archive layout and update the config if the competition changed.')
    print(f'Labels: {csv_path}')
    print(f'Images: {image_dir} (verify this directory exists; competition archives may require separate image archives)')
    return csv_path


def combine(specs, output):
    """Combine datasets into one canonical CSV; IDs are paths relative to data/.

    spec syntax: name|labels.csv|image_dir|id_column|label_column
    Example: aptos|data/aptos/train.csv|data/aptos/train_images|id_code|diagnosis
    """
    rows, seen = [], set()
    data_root = DATA.resolve()
    for spec in specs:
        p = spec.split('|')
        if len(p) != 5 or not all(p):
            raise ValueError(f'Invalid spec {spec!r}; expected name|labels.csv|image_dir|id_column|label_column')
        name, csv_path, image_dir, id_col, label_col = p
        csv_path, image_dir = Path(csv_path).resolve(), Path(image_dir).resolve()
        if not csv_path.is_file() or not image_dir.is_dir():
            raise FileNotFoundError(f'{name}: label CSV or image directory missing: {csv_path} / {image_dir}')
        try:
            image_dir.relative_to(data_root)
        except ValueError as e:
            raise ValueError(f'{name}: image directory must be inside {data_root} so paths remain portable') from e
        df = pd.read_csv(csv_path)
        if id_col not in df or label_col not in df:
            raise ValueError(f'{name}: expected columns {id_col!r}, {label_col!r}; found {list(df.columns)}')
        for _, row in df.iterrows():
            ident = str(row[id_col]).strip()
            try:
                grade = int(row[label_col])
            except (ValueError, TypeError):
                raise ValueError(f'{name}: non-integer label for image {ident!r}')
            if grade not in range(5):
                raise ValueError(f'{name}: grade must be 0..4; got {grade} for {ident!r}')
            candidates = [image_dir / ident, image_dir / f'{ident}.png', image_dir / f'{ident}.jpg', image_dir / f'{ident}.jpeg', image_dir / f'{ident}.JPG', image_dir / f'{ident}.PNG']
            image = next((x for x in candidates if x.is_file()), None)
            if image is None:
                raise FileNotFoundError(f'{name}: image not found for ID {ident!r} in {image_dir}')
            rel = image.resolve().relative_to(data_root).as_posix()
            # Prefix IDs by dataset in the output to avoid collisions, keep actual path in a private helper column.
            key = (name, ident)
            if key in seen:
                raise ValueError(f'{name}: duplicate image ID {ident!r}')
            seen.add(key)
            rows.append({'image': rel, 'level': grade, 'source_dataset': name})
    if not rows:
        raise ValueError('No images found to combine')
    out = Path(output)
    if not out.is_absolute(): out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Wrote {len(rows)} labelled rows to {out}')
    print('For training, use the project data folder as image_dir. Example: python train.py --train "combined|data/combined_train.csv|data"')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download', choices=sorted(DATASETS), help='Download an accessible Kaggle competition dataset')
    ap.add_argument('--combine', nargs='+', metavar='SPEC', help='Combine two or more datasets using name|labels.csv|image_dir|id_column|label_column')
    ap.add_argument('--output', default='data/combined_train.csv')
    a = ap.parse_args()
    if a.download:
        download(a.download)
    elif a.combine:
        combine(a.combine, a.output)
    else:
        ap.print_help()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(2)
