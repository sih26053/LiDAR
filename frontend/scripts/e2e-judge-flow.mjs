/**
 * Judge-flow E2E (STEP 42): drives the REAL dashboard in headless Chrome
 * against the LIVE local backend, exactly as a judge would:
 * select frame -> Run -> map/metrics/benchmark appear -> screenshot.
 *
 * Run:  node scripts/e2e-judge-flow.mjs [frameId]
 * Requires: backend on :8000, frontend (preview/dev) on :5173.
 */
import puppeteer from 'puppeteer-core';

const WEB = process.env.E2E_WEB_URL || 'http://127.0.0.1:5173/';
const FRAME = process.argv[2] || '5991fad3280c4f84b331536c32001a04';
const SHOT = process.env.E2E_SHOT || 'C:/Users/SARKAR/AppData/Local/Temp/opencode/judge-flow.png';

const results = [];
const check = (name, ok, detail = '') => {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}${detail ? ' -- ' + detail : ''}`);
};

const browser = await puppeteer.launch({
  executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu', '--window-size=1500,2200'],
});
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1500, height: 2200 });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));

  await page.goto(WEB, { waitUntil: 'networkidle0', timeout: 60000 });
  check('dashboard loads', true);

  // Backend connected (explicit, never assumed)
  await page.waitForFunction(
    () => document.body.textContent.includes('Backend: Connected'),
    { timeout: 30000 },
  );
  check('backend connected banner', true);

  // Frames populate the selector from GET /frames
  await page.waitForFunction(
    () => document.querySelectorAll('select option').length > 1,
    { timeout: 30000 },
  );
  const options = await page.$$eval('select option', (els) => els.map((e) => e.value).filter(Boolean));
  check('frame selector populated from backend', options.includes(FRAME), `${options.length} frames`);

  // Select the real frame
  await page.select('select', FRAME);
  check('frame selection works', true, FRAME.slice(0, 8));

  // Run (real POST /replay/run through the UI)
  await page.evaluate(() => {
    const btn = [...document.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Run');
    btn?.click();
  });
  await page.waitForFunction(
    () => [...document.querySelectorAll('.badge')].some((b) => b.textContent.includes('READY') && !document.body.textContent.includes('No result available for this frame. Select a frame')),
    { timeout: 300000 },
  ).catch(() => {});
  const bodyText = await page.evaluate(() => document.body.textContent);

  check('original LiDAR renders', bodyText.includes('Replayed input') && bodyText.includes('cells shown'));
  check('adaptive map renders', bodyText.includes('marker size = backend resolution'));
  check('importance view renders', bodyText.includes('importance » resolution decision'));
  check('metrics update (measured)', bodyText.includes('Mapping latency') && bodyText.includes('ms'));
  check('semantic source shown (not AI prediction)', /Semantic Source:/.test(bodyText) && !bodyText.includes('AI prediction'));
  check('benchmark stored results shown', bodyText.includes('Stored benchmark result'));
  check('current replay row shown', bodyText.includes('Current replay result'));
  check('no fake-data leakage', !bodyText.includes('NaN') && errors.length === 0, errors.slice(0, 2).join(' | '));

  await page.screenshot({ path: SHOT.replace('.png', '-with-data.png'), fullPage: false });
  console.log('screenshot (with data):', SHOT.replace('.png', '-with-data.png'));

  // Navigate to another frame and confirm panels reset to the new frame (no stale mix)
  const other = options.find((o) => o !== FRAME);
  if (other) {
    await page.select('select', other);
    const afterNav = await page.evaluate(() => document.body.textContent);
    check('frame navigation resets panels (no stale mix)', afterNav.includes('No result available for this frame'));
  }

  await page.screenshot({ path: SHOT, fullPage: false });
  console.log('screenshot:', SHOT);
} finally {
  await browser.close();
}
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} judge-flow checks passed`);
process.exit(failed.length ? 1 : 0);
