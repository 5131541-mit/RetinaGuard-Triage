#!/usr/bin/env bash
# End-to-end smoke test on SYNTHETIC data (minutes on CPU). Real results need APTOS/IDRiD: see README.
set -e
python make_demo_data.py --n 200 --out data/demo
python train.py --train "demo|data/demo/train.csv|data/demo/images" --epochs 2 --size 256 --bs 8 --workers 0 --no-pretrained
uvicorn server:app --port 8000
