# Release readiness checklist

**Release type:** research prototype / engineering candidate, not a clinically validated diagnostic system.

## Dataset verification
- [ ] APTOS, EyePACS, IDRiD and DDR download/access tested with valid credentials and accepted terms
- [x] Label range/column checks, missing image checks, corrupt image checks and duplicate ID checks implemented
- [x] Exact duplicate content and perceptual-hash near-duplicate screening implemented
- [x] Patient-grouped splits used when recognized patient IDs exist; splits fail closed if any grade is absent
- [ ] Real datasets have been acquired and the label mapping for each source reviewed
- [ ] Near-duplicate screen manually reviewed on real dataset(s); false positives/negatives documented
- [ ] APTOS patient leakage limitation addressed or explicitly accepted for exploratory evaluation

## Model verification
- [x] Missing or invalid model weights disable inference instead of generating a prediction
- [x] Checkpoint loading uses weights-only loading, strict state dict validation and input-size validation
- [x] Checkpoint SHA-256 and readiness status are exposed by the health endpoint
- [x] Metrics include per-grade sensitivity/specificity, macro-F1, QWK, confusion matrix, referable AUROC and referable-DR ECE
- [ ] Training completed on real labelled retinal images
- [ ] Locked held-out test set evaluated once and reported
- [ ] Calibration and uncertainty evaluated on real held-out data
- [ ] Image-quality rejection and out-of-distribution behavior evaluated on representative data
- [ ] Severe/proliferative DR false negatives reviewed by qualified experts

## API and frontend
- [x] API-level tests cover state persistence sanitization, upload validation, PDF generation, dataset status and checkpoint rejection
- [x] Training paths are restricted to configured data root
- [x] Upload byte size, decoded dimensions and pixel count are checked
- [x] Server-side UI state strips image payloads
- [x] Playwright browser E2E suite added
- [ ] Playwright suite executed on a supported local browser
- [ ] All six UI sections and every interaction manually verified
- [ ] Desktop/mobile responsive behavior and screenshot regression reviewed
- [ ] End-to-end workflow tested with a verified trained checkpoint

## Security and deployment
- [x] Local-only API by default; bearer-token check available when `RG_API_TOKEN` is configured
- [x] Audit logging avoids upload filenames/raw image bytes
- [x] SQLite WAL and busy timeout configured
- [x] Release documentation explains security boundaries and deployment requirements
- [ ] Production-grade user identity and per-case authorization implemented
- [ ] HTTPS, reverse-proxy/SSO, rate limiting, backups, retention, monitoring and incident response configured
- [ ] Threat model and privacy/security review completed
- [ ] Multi-user data isolation implemented before handling multiple users' records

## Clinical and research validation
- [ ] Independent external dataset evaluation completed
- [ ] Lesion localization validated against expert annotations; current lesion candidates remain heuristic
- [ ] Calibration, subgroup analysis and site/camera shift reviewed
- [ ] Intended use, limitations and human-review requirements approved
- [ ] Clinical, ethics and regulatory review completed as applicable

## Release packaging
- [x] Pinned runtime dependencies and pytest dependency included
- [x] Verification report, security notes, model-card template and browser E2E plan included
- [x] Release archive excludes local `app.db`, Python caches and pytest cache
- [x] Python compilation, 25 automated tests, frontend JS syntax and E2E JS syntax checks passed in the build workspace
- [ ] Full clean-install test on a separate machine/virtual environment completed
- [ ] Browser E2E and real-data training evidence attached to release record

## Known blockers
- No APTOS/EyePACS/IDRiD/DDR dataset is bundled or verified as downloaded.
- No trained checkpoint or real-data metrics are bundled.
- Browser E2E suite was authored but not run in this restricted browser environment.
- The API token is single-operator protection, not full multi-user authentication. Public deployment requires a proper auth gateway and authorization design.
- PDF report fields are client-supplied and not cryptographically or server-record verified.
- Clinical validation is not complete; this project must not be used for diagnosis or treatment decisions.
