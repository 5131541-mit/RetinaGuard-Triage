# Browser end-to-end test plan

Browser E2E tests were **not executed in the build environment**. Run them on a workstation with a supported Chromium browser and compare screenshots at the same viewport as `reference_screenshots/`.

## Required scenarios
1. Load dashboard and confirm `/api/health` renders `missing_weights` without a crash.
2. Navigate through Dashboard, Image Analysis, Heatmap Analysis, Datasets & Models, Reports and Settings; verify active navigation and no console errors.
3. Upload an unsupported file, a corrupt image, an oversized image and a valid-but-poor-quality image; confirm a clear error and no fabricated result.
4. With a verified trained checkpoint, upload a valid fundus image and verify grade/probabilities, quality gate, uncertainty, model hash, and research disclaimer are displayed.
5. Save settings/history, restart server, and verify metadata persists while uploaded image bytes are not written to SQLite.
6. Generate a PDF report and verify it contains the research-prototype disclaimer and clearly identifies model/result metadata as unverified client-supplied fields unless the report API is redesigned to retrieve a server-stored analysis.
7. Verify dataset status, training start/status/error handling, checkpoint upload rejection, export-without-weights rejection and audit access.
8. With `RG_API_TOKEN` set, confirm the dashboard prompts once, successful token entry allows API use, wrong token receives 401, and tokenless remote requests are denied.
9. Test narrow/mobile viewport, slow network, API restart during upload, repeated clicks and keyboard accessibility.

## Acceptance criteria
- No JavaScript console errors during normal navigation.
- No demo/sample record is presented as a real inference.
- No missing/invalid model can produce a prediction.
- Every error has a visible recovery message.
- No image bytes or secrets appear in server-side state or audit logs.
- Screenshot differences are reviewed, not automatically ignored.
