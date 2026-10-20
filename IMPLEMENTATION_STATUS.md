# RetinaGuard implementation status

## Implemented in this release candidate
- Existing six-page dark teal dashboard retained.
- FastAPI backend and SQLite persistence with WAL/busy timeout.
- Explicit model readiness states and fail-closed inference when weights are missing or invalid.
- Safer checkpoint loading (`weights_only=True`, strict state dict, input size validation) and SHA-256 model fingerprint.
- Upload byte-size, decoded image, minimum-dimension and pixel-count validation plus image-quality gate.
- Training inputs restricted to `RG_DATA_ROOT` after real-path resolution.
- Dataset validation for labels, missing/corrupt files, duplicate IDs, exact duplicate image content and heuristic perceptual-hash near-duplicates.
- Patient-level splits when patient IDs are present; otherwise an explicit image-level split limitation.
- Evaluation metrics include per-grade sensitivity/specificity, macro-F1, QWK, confusion matrix, referable AUROC and referable-DR ECE.
- Server-side UI state strips image payloads before SQLite persistence.
- API token enforcement option, local-only default, and audit messages that avoid upload filenames/raw image content.
- Pinned dependencies, 25 automated tests, security notes, model-card template, release checklist and browser E2E test suite.

## Still not completed
- Real dataset acquisition and label-map review.
- Real-data model training, locked held-out evaluation and independent external validation.
- Browser E2E execution and screenshot comparison.
- Production-grade user identity/per-case authorization, multi-user database isolation and hardened public deployment.
- Clinical review and lesion segmentation validation. Lesion candidates remain heuristic.
- PDF reports still format client-supplied result fields rather than retrieving an immutable server-side analysis record.

## Responsible use
This is a research prototype. Do not use it to diagnose or make patient-care decisions. Automated software tests are not evidence of diagnostic accuracy.
