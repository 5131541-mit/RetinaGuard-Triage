# Browser E2E tests

Run from the project root after installing Python dependencies:

```bash
cd e2e
npm install
npx playwright install chromium
npm test
```

The Playwright suite was added as an executable test suite, but it was **not run in the build environment** because browser execution was restricted there. Run it on a workstation with Chromium installed and review any visual differences against `../reference_screenshots/`.
