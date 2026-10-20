const { test, expect } = require('@playwright/test');

test('dashboard serves and API explicitly reports missing model weights', async ({ page, request }) => {
  const health = await request.get('/api/health');
  expect(health.ok()).toBeTruthy();
  const h = await health.json();
  expect(h).toHaveProperty('model_loaded');
  if (!h.model_loaded) expect(h.model_status).toMatch(/missing_weights|invalid_checkpoint/);
  await page.goto('/');
  await expect(page.locator('body')).toContainText('RetinaGuard');
  await expect(page.locator('body')).toContainText('Dashboard');
});

test('main navigation opens each primary page', async ({ page }) => {
  await page.goto('/');
  for (const label of ['Image Analysis', 'Patient Records', 'Reports', 'Dataset & Models', 'Settings']) {
    const nav = page.getByText(label, { exact: true }).first();
    await expect(nav).toBeVisible();
    await nav.click();
    await expect(page.locator('body')).toContainText(label);
  }
});

test('state persistence excludes image payloads', async ({ request }) => {
  const put = await request.put('/api/state', { data: { recs: [{ id: 'e2e-case', img: 'data:image/png;base64,PRIVATE', grade: 2 }], set: { theme: 'dark' } } });
  expect(put.ok()).toBeTruthy();
  const saved = await request.get('/api/state');
  const body = await saved.json();
  expect(JSON.stringify(body)).not.toContain('PRIVATE');
  expect(body.recs[0]).not.toHaveProperty('img');
});
