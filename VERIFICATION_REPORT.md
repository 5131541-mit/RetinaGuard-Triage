# RetinaGuard Triage — verification report

Verification date: 2026-10-20

## Automated checks completed in this workspace

- `python -m py_compile server.py train.py lesion_eval.py` — passed.
- `python -m pytest -q` — **25 passed**.
- Frontend inline JavaScript `node --check` — passed.
- Playwright configuration and test JavaScript syntax checks — passed.
- Uvicorn started on localhost; `GET /` returned HTTP 200.
- `GET /api/health` returned `model_status=missing_weights`, `model_loaded=false`, and no checkpoint hash when no trained model was present.
- API tests cover state persistence sanitization, access-token enforcement, upload validation, PDF generation, dataset status, training path restriction and invalid checkpoint rejection.
- Dataset validation tests cover labels, missing files, corrupt files, duplicate content and insufficient class support.
- Release archive integrity was verified after packaging; local database and generated caches are excluded.

## Code safeguards added

- Fail-closed model loading with `weights_only=True`, strict state dictionary validation, supported input-size validation, readiness status and checkpoint SHA-256.
- Training input paths restricted to the configured data root.
- Dataset validation includes image decoding, exact duplicate detection and a perceptual-hash near-duplicate screening heuristic.
- Patient-grouped train/validation/test splitting when recognized patient identifiers are available; fails closed if grade coverage is inadequate.
- Per-grade sensitivity and specificity added to the evaluation metrics. ECE is explicitly scoped to binary referable DR, not multiclass calibration.
- API upload limits include byte size, image dimensions and pixel count.
- Server-side UI state persistence excludes image payloads; audit messages avoid raw images and uploaded filenames.
- API routes are local-only by default. A configured `RG_API_TOKEN` enables bearer-token enforcement; the browser asks for the operator token when needed.
- Pinned dependencies, security documentation, model card template and a Playwright E2E suite were added.

## Not verified / release blockers

- No real APTOS, EyePACS, IDRiD or DDR dataset is present; dataset acquisition, terms, actual labels and mappings remain unverified.
- No real-data training run was completed and no real performance metrics or trained checkpoint are bundled.
- Browser E2E tests were authored but not executed because browser execution was restricted in this environment. JS syntax checks are not a substitute for browser testing.
- Patient-level separation cannot be guaranteed for datasets without patient IDs (including standard APTOS labels). Perceptual-hash screening is heuristic and does not prove absence of all near-duplicates.
- No validated lesion segmentation model exists; current lesion markers are heuristic candidates.
- Token protection is not multi-user authentication/authorization. SQLite state is one shared UI state. PDF reports are based on client-supplied result fields.
- No clinical validation, regulatory assessment or patient-care safety review has been performed.

## Reproduce checks

```bash
python -m py_compile server.py train.py lesion_eval.py
python -m pytest -q
```

## Local run

```bash
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

Without valid trained weights, the API must not produce a prediction. This remains a research prototype, not a diagnostic system.
