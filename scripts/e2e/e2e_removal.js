// Phase 15: the removal engines on the live stack (API, Redis, Celery worker, S3 stand-in, database, built web app).
// Start the stack with AI_PROVIDER=auto, run make_removal_media.sh, then:  run.sh e2e_removal.js
const { chromium } = require("playwright");
const { execSync } = require("child_process");
const path = require("path");
const { DIR, VENV, media } = require("./lib");
const WEB = "http://localhost:3000", API = "http://localhost:8000";
let failures = 0;
async function step(name, fn) {
  try { const d = await fn(); console.log(`PASS  ${name}${d ? "  — " + d : ""}`); }
  catch (e) { failures += 1; console.log(`FAIL  ${name}  — ${String(e.message).split("\n")[0].slice(0, 400)}`); }
}
const assert = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const PASSWORD = "correct horse battery";
const api = (page, p, opts) => page.evaluate(async ([a, p, opts]) => { const r = await fetch(a + p, { credentials: "include", ...(opts || {}) }); let body = null; try { body = await r.json(); } catch {} return { status: r.status, body }; }, [API, p, opts]);
const json = (body) => ({ headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
const rect = (x, y, w, h) => ({ version: 1, regions: [{ id: "r", name: "Logo", type: "rectangle", keyframes: [{ frame: 0, rect: { x, y, w, h } }] }] });

async function upload(page, file) {
  await page.goto(`${WEB}/dashboard`);
  await page.setInputFiles("input[type=file]", path.join(media, file));
  await page.getByRole("button", { name: "Upload video" }).click();
  await page.getByText("Video ready").first().waitFor({ timeout: 90000 });
  await page.getByRole("link", { name: "Open editor" }).first().click();
  await page.waitForURL("**/editor/*");
  return page.url().split("/editor/")[1];
}

(async () => {
  const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 1100 }, acceptDownloads: true });
  const page = await ctx.newPage();
  const email = `ada${Date.now()}@example.com`;
  await page.goto(`${WEB}/signup`);
  await page.fill("#name", "Ada Lovelace"); await page.fill("#email", email); await page.fill("#password", PASSWORD);
  await page.click("button[type=submit]");
  await page.waitForURL("**/dashboard", { timeout: 15000 });

  await step("the engine status reports the automatic engine as real, not development", async () => {
    const r = await api(page, "/api/engine");
    assert(r.status === 200 && r.body.provider === "auto" && r.body.development_only === false && r.body.available === true, JSON.stringify(r.body));
    return `${r.body.name}: ${r.body.expectation.slice(0, 90)}`;
  });

  let id;
  await step("a semi-transparent logo is removed through the real worker, and the notes say how", async () => {
    id = await upload(page, "overlay.mp4");
    // The logo box in the clip is (24,36)-(282,124) of 320x160.
    const saved = await api(page, `/api/projects/${id}/mask`, { method: "PUT", ...json(rect(24 / 320, 36 / 160, 258 / 320, 88 / 160)) });
    assert(saved.status === 200, "mask save " + saved.status + JSON.stringify(saved.body));
    await page.reload();
    await page.getByRole("button", { name: "Remove marked areas" }).click();
    await page.getByText("Finished and checked").waitFor({ timeout: 180000 });
    const { job } = (await api(page, `/api/projects/${id}/render`)).body;
    assert(job.status === "completed" && job.engine.provider === "auto" && job.engine.development_only === false, JSON.stringify(job.engine));
    assert(job.result.frames === 60 && job.result.audio === "copied", JSON.stringify(job.result));
    const note = (job.warnings || []).find((w) => /inverted/.test(w));
    assert(note, "warnings: " + JSON.stringify(job.warnings));
    return note;
  });

  await step("the result page shows that note, no 'not AI / blur' notice, and the download matches the true clean video", async () => {
    await page.getByRole("link", { name: "Preview and download" }).click();
    await page.getByRole("heading", { name: "Your restored video is ready." }).waitFor({ timeout: 15000 });
    await page.getByText(/measured from the video itself and inverted/).first().waitFor({ timeout: 5000 });
    assert((await page.getByText(/not an? AI/i).count()) === 0, "the development notice must not appear for a real engine");
    const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: /Download MP4/ }).click()]);
    const file = path.join(DIR, "removal-result.mp4");
    await download.saveAs(file);
    const out = execSync(`${VENV}/bin/python - <<'PY'
import subprocess, numpy as np
W,H=320,160
def frames(p):
    raw=subprocess.run(["ffmpeg","-v","error","-i",p,"-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True,check=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(-1,H,W,3).astype(float)
src,res,truth=frames("${path.join(media, "overlay.mp4")}"),frames("${file}"),frames("${path.join(media, "overlay_truth.mp4")}")
assert len(res)==len(truth)==60,(len(res),len(truth))
region=np.zeros((H,W),bool); region[40:120,30:276]=True
def psnr(a,b): return 10*np.log10(255**2/((a[:,region]-b[:,region])**2).mean())
print(f"{psnr(src,truth):.1f} {psnr(res,truth):.1f}")
PY`).toString().trim().split(" ").map(Number);
    assert(out[1] > 28 && out[1] > out[0] + 10, `PSNR in the logo area: before ${out[0]} dB, after ${out[1]} dB`);
    return `PSNR in the marked area against the clean footage: ${out[0]} dB with the logo -> ${out[1]} dB restored`;
  });

  await step("a fixed opaque logo on a still camera fails with the real reason, shown in the editor, marks untouched", async () => {
    const id2 = await upload(page, "fixed.mp4");
    // The mark is (230,140)-(258,158) of 320x200.
    const saved = await api(page, `/api/projects/${id2}/mask`, { method: "PUT", ...json(rect(230 / 320, 140 / 200, 28 / 320, 18 / 200)) });
    assert(saved.status === 200, "mask save " + saved.status);
    await page.reload();
    await page.getByRole("button", { name: "Remove marked areas" }).click();
    await page.getByText(/hidden in every nearby frame/).first().waitFor({ timeout: 180000 });
    const { job } = (await api(page, `/api/projects/${id2}/render`)).body;
    assert(job.status === "failed" && !job.result, JSON.stringify({ s: job.status, r: job.result }));
    const mask = await api(page, `/api/projects/${id2}/mask`);
    const kept = mask.body.mask_data;
    assert(mask.status === 200 && kept.regions.length === 1, "marks must be kept: " + JSON.stringify(mask.body).slice(0, 200));
    return job.error.slice(0, 140);
  });

  await step("a logo that fades in and out is removed, the notes say so, and the measuring step is a real stage", async () => {
    const id3 = await upload(page, "fade.mp4");
    const saved = await api(page, `/api/projects/${id3}/mask`, { method: "PUT", ...json(rect(24 / 320, 36 / 160, 258 / 320, 88 / 160)) });
    assert(saved.status === 200, "mask save " + saved.status);
    await page.reload();
    await page.getByRole("button", { name: "Remove marked areas" }).click();
    const seen = new Set();
    let job = null;
    for (let i = 0; i < 900; i += 1) {  // poll the job like the interface does, and note every stage it really reports
      job = (await api(page, `/api/projects/${id3}/render`)).body.job;
      if (job) { seen.add(job.stage); if (["completed", "failed", "cancelled"].includes(job.status)) break; }
      await page.waitForTimeout(200);
    }
    assert(job && job.status === "completed", JSON.stringify(job && { s: job.status, e: job.error }));
    const stages = Object.fromEntries(job.stages.map((s) => [s.key, s.state]));
    assert(stages.measuring === "done", "stages: " + JSON.stringify(stages));
    assert((job.warnings || []).some((w) => /strength changed/.test(w)), "warnings: " + JSON.stringify(job.warnings));
    const file = path.join(DIR, "fade-result.mp4");
    await page.goto(`${WEB}/result/${id3}`);
    await page.getByRole("heading", { name: "Your restored video is ready." }).waitFor({ timeout: 15000 });
    const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: /Download MP4/ }).click()]);
    await download.saveAs(file);
    const out = execSync(`${VENV}/bin/python - <<'PY'
import subprocess, numpy as np
W,H=320,160
def frames(p):
    raw=subprocess.run(["ffmpeg","-v","error","-i",p,"-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True,check=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(-1,H,W,3).astype(float)
src,res,truth=frames("${path.join(media, "fade.mp4")}"),frames("${file}"),frames("${path.join(media, "fade_truth.mp4")}")
assert len(res)==len(truth)==72,(len(res),len(truth))
region=np.zeros((H,W),bool); region[40:120,30:276]=True
def psnr(a,b): return 10*np.log10(255**2/((a[:,region]-b[:,region])**2).mean())
print(f"{psnr(src,truth):.1f} {psnr(res,truth):.1f}")
PY`).toString().trim().split(" ").map(Number);
    assert(out[1] > 26 && out[1] > out[0] + 6, `PSNR in the logo area: before ${out[0]} dB, after ${out[1]} dB`);
    return `stages seen: ${[...seen].join(" > ")}; PSNR ${out[0]} dB -> ${out[1]} dB`;
  });

  await browser.close();
  console.log(failures ? `\n${failures} step(s) FAILED` : "\nall steps passed");
  process.exit(failures ? 1 : 0);
})();
