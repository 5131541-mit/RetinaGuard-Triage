# RetinaGuard Triage — full-stack research prototype

**Not a medical device. Not validated for diagnosis or patient-care decisions.** Model outputs, Grad-CAM maps, uncertainty estimates and heuristic lesion candidates are research aids only and must not be used to make clinical decisions.

## Architecture and safeguards

- FastAPI backend serves the six-page dashboard from `frontend/index.html`.
- Model loading uses PyTorch `weights_only=True`, strict state-dictionary matching, an allowed input-size list and a SHA-256 checkpoint fingerprint. Missing or invalid weights disable inference; the API does not generate a fallback/demo prediction.
- Image upload checks content decoding, dimensions, pixel count and byte size before quality assessment.
- Training is separate from prediction and can read only CSV/image paths inside `RG_DATA_ROOT` (default `./data`).
- Training checks labels, missing/corrupt files, duplicate IDs, exact duplicate content and perceptual-hash near-duplicate candidates. It uses patient-grouped splits when a recognized patient identifier exists; otherwise it reports image-level split limitations.
- Evaluation reports QWK, macro-F1, confusion matrix, per-grade sensitivity/specificity, referable-DR AUROC and binary referable-DR ECE. ECE is not a multiclass calibration guarantee.
- SQLite uses WAL and a busy timeout. UI metadata/settings can persist; image payloads are deliberately stripped from server-side state storage.
- Audit events avoid storing uploaded filenames or image contents. They are append-only through the exposed API (direct database administrators can still modify the SQLite file).
- The API is local-only by default. Set `RG_API_TOKEN` for bearer-token protection of API routes. The dashboard prompts for that token and holds it in session storage for the current browser tab. This is a single-operator prototype mechanism, not multi-user identity management; use HTTPS plus a trusted reverse proxy/SSO and per-user authorization before any public deployment.

## 1. Local installation

Python 3.11 is the tested target. Create a virtual environment, then:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Without `outputs/best.pt`, `/api/health` reports `missing_weights` and `/api/analyze` returns HTTP 503 after validating the image. This is expected until a real model checkpoint has been trained and evaluated.

### Protected single-operator mode

Generate a long random token and keep it secret:

```bash
# macOS/Linux
export RG_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uvicorn server:app --host 127.0.0.1 --port 8000
```

For Windows PowerShell:

```powershell
$env:RG_API_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
uvicorn server:app --host 127.0.0.1 --port 8000
```

The dashboard asks for the token when an API call returns 401. Do not expose the shared token or use this mechanism as a substitute for real user authentication. For remote deployment, put the app behind HTTPS and a reverse proxy that performs user authentication and safely injects the API credential; do not expose Uvicorn directly to the public internet.

## 2. Dataset acquisition

Datasets are **not included**. Download only after accepting each provider's terms; do not redistribute restricted images.

### APTOS / EyePACS

1. Install Kaggle CLI: `python -m pip install kaggle`.
2. Configure credentials according to [Kaggle API instructions](https://github.com/Kaggle/kaggle-api#api-credentials). Never commit `kaggle.json`.
3. Accept the competition rules at [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) and/or [EyePACS](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data).
4. Run `python prepare_datasets.py --download aptos` or `python prepare_datasets.py --download eyepacs` and verify the resulting files manually.

IDRiD and DDR must be obtained through their official pages/repositories and according to their access terms. Label columns differ by release; inspect and map them deliberately. Do not combine an external evaluation dataset into training and then describe it as independent testing.

## 3. Train and evaluate

Place data under `data/`, for example `data/aptos/train.csv` and `data/aptos/train_images/`. From the project root:

```bash
python train.py --train "aptos|data/aptos/train.csv|data/aptos/train_images" --epochs 15 --size 384 --bs 8 --workers 2 --out outputs
```

The program writes `outputs/best.pt`, `outputs/metrics.json`, `outputs/confusion_matrix.png`, `outputs/errors.csv`, split manifests and training history. A CUDA GPU is recommended for EfficientNet-B3. The program refuses datasets with invalid labels, missing/corrupt files, duplicate image content, possible perceptual-hash near-duplicates, or insufficient class support. Near-duplicate checks are heuristic and do not prove the absence of all leakage.

If patient IDs are present under a recognized column name, patient-grouped splitting is used and training stops if any split lacks a grade. APTOS does not provide patient IDs in its standard label file, so patient-level leakage cannot be ruled out. This limitation must be stated with any reported metrics.

Use `--ext "idrid|data/idrid/labels.csv|data/idrid/images"` only for a dataset that was not used for training/tuning. Verify grade semantics and labels first. Training and evaluation must be repeated on the actual intended datasets; no real data or clinical metrics are included in this ZIP.

## 4. API routes

- `GET /api/health` — model readiness, checkpoint SHA-256 and training state.
- `POST /api/analyze` — validated image inference; unavailable model returns 503.
- `GET|PUT /api/state` — UI metadata/settings; image payloads are excluded from server-side persistence.
- `GET /api/audit` — recent audit events.
- `GET /api/dataset/status`, `GET /api/metrics` — local artifact status and training metrics.
- `POST /api/train`, `GET /api/train/status` — local training job; inputs must be under `RG_DATA_ROOT`.
- `POST /api/model/upload`, `POST /api/model/export` — trusted checkpoint upload and ONNX export.
- `POST /api/report` — PDF research summary of client-supplied result fields; it does not verify that those fields came from a stored inference and must not be treated as a trusted clinical record.

Interactive API docs are at `/docs`. Do not expose the API docs or training/model-upload operations publicly without a hardened deployment design.

## 5. Verification

```bash
python -m py_compile server.py train.py lesion_eval.py
python -m pytest -q
```

The automated suite checks API response behavior, persistence sanitization, access-token enforcement, report generation, training path restrictions, dataset label validation, corrupt/missing files and duplicate-image checks. It does **not** establish model accuracy or clinical validity.

### Remaining release blockers

- Real dataset download/access and label mappings have not been verified here.
- No real-data training run or held-out/independent clinical performance results are included.
- Browser E2E/screenshot regression must be run in an environment with a supported browser; backend tests do not prove all UI interactions work.
- Heuristic lesion candidates are not a trained/validated lesion-segmentation model.
- The single-operator token mechanism is not a multi-user authentication/authorization system.
- Clinical validation, intended-use review, risk management, privacy review and applicable regulatory review remain outstanding.
