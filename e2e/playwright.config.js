const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
  testDir: '.',
  timeout: 30000,
  use: { baseURL: 'http://127.0.0.1:8000', headless: true, viewport: { width: 1440, height: 900 } },
  webServer: {
    command: 'python -m uvicorn server:app --host 127.0.0.1 --port 8000',
    url: 'http://127.0.0.1:8000/api/health',
    reuseExistingServer: !process.env.CI,
    timeout: 120000
  }
});
