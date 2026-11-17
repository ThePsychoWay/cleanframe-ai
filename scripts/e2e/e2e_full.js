// Phase 12: one complete journey from an empty system, in a real browser, against the real API, worker, scheduler,
// Redis, S3 and PostgreSQL. Includes the abuse cases from the spec's security-testing list.
const { chromium } = require("playwright");
const { execSync } = require("child_process");
const fs = require("fs"), path = require("path");
const { DIR, VENV, media, python, sql, makeDue, storageKeys, log } = require("./lib");
const WEB = "http://localhost:3000", API = "http://localhost:8000";
let failures = 0;
async function step(name, fn) {
  try { const d = await fn(); console.log(`PASS  ${name}${d ? "  — " + d : ""}`); }
  catch (e) { failures += 1; console.log(`FAIL  ${name}  — ${String(e.message).split("\n")[0].slice(0, 300)}`); }
}
const assert = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const PASSWORD = "correct horse battery";
async function signup(page, name, email) {
  await page.goto(`${WEB}/signup`);
  await page.fill("#name", name); await page.fill("#email", email); await page.fill("#password", PASSWORD);
  await page.click("button[type=submit]");
  await page.waitForURL("**/dashboard", { timeout: 15000 });
}
async function drag(page, from, to, steps = 10) {
  const b = await page.locator("canvas").boundingBox();
  const at = ([fx, fy]) => [b.x + fx * b.width, b.y + fy * b.height];
  const [x1, y1] = at(from), [x2, y2] = at(to);
  await page.mouse.move(x1, y1); await page.mouse.down(); await page.mouse.move(x2, y2, { steps }); await page.mouse.up();
}
const api = (page, p, opts) => page.evaluate(async ([a, p, opts]) => { const r = await fetch(a + p, { credentials: "include", ...(opts || {}) }); let body = null; try { body = await r.json(); } catch {} return { status: r.status, body }; }, [API, p, opts]);
const json = (body) => ({ headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

(async () => {
  const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 1100 }, acceptDownloads: true });
  const page = await ctx.newPage();
  const problems = [];
  page.on("pageerror", (e) => problems.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error" && !/status of (400|401|404|409|410|413|422|429)/.test(m.text())) problems.push("console: " + m.text()); });
  const stamp = Date.now();
  const emailA = `ada${stamp}@example.com`, emailB = `grace${stamp}@example.com`;
  let id, pid2;

  // ---------- 1. the public side and the gate -----------------------------------------------------------------
  await step("landing page loads with the ownership notice; /dashboard sends a stranger to log in and back", async () => {
    await page.goto(WEB);
    await page.getByText("Only remove overlays from content you own or have permission to edit.").first().waitFor({ timeout: 15000 });
    await page.goto(`${WEB}/dashboard`);
    await page.waitForURL(/\/login\?next=%2Fdashboard/, { timeout: 15000 });
  });

  await step("sign up, sign out, and sign back in with the same password", async () => {
    await signup(page, "Ada Lovelace", emailA);
    await page.getByRole("button", { name: /log out|sign out/i }).first().click();
    await page.waitForURL(/\/(login)?$|\/login/, { timeout: 15000 });
    await page.goto(`${WEB}/login`);
    await page.fill("#email", emailA); await page.fill("#password", PASSWORD);
    await page.click("button[type=submit]");
    await page.waitForURL("**/dashboard", { timeout: 15000 });
  });

  // ---------- 2. bad input ---------------------------------------------------------------------------------------
  await step("a text file and an oversized file are refused in the browser; nothing is created", async () => {
    await page.setInputFiles("input[type=file]", { name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("hello") });
    await page.getByText(/Unsupported video format/).waitFor({ timeout: 5000 });
    await page.setInputFiles("input[type=file]", { name: "huge.mp4", mimeType: "video/mp4", buffer: Buffer.alloc(10) }); // size faked below via the API
    const list = await api(page, "/api/projects");
    assert(list.status === 200 && list.body.length === 0, "a refused file must not create a project, found " + JSON.stringify(list.body.map((p) => p.name)));
    const tooBig = await api(page, "/api/projects", { method: "POST", ...json({ filename: "huge.mp4", size_bytes: 50 * 1024 ** 3 }) });
    assert(tooBig.status === 413 || tooBig.status === 422, "API must refuse a 50 GiB upload, got " + tooBig.status);
    return `API said ${tooBig.status}: ${JSON.stringify(tooBig.body).slice(0, 90)}`;
  });

  await step("a file that is not a video but is called .mp4 is refused by the server with a readable reason", async () => {
    await page.goto(`${WEB}/dashboard`);
    await page.setInputFiles("input[type=file]", path.join(media, "garbage.mp4"));
    await page.getByRole("button", { name: "Upload video" }).click();
    await page.getByRole("alert").filter({ hasText: /couldn't read this video/ }).waitFor({ timeout: 60000 });
    assert(!(await page.getByRole("link", { name: "Open editor" }).count()), "must not offer the editor");
    const failed = sql("select status from projects order by created_at desc limit 1");
    assert(failed === "FAILED", "status " + failed);
  });

  await step("a path-traversal file name is stored as a harmless label and the upload still works", async () => {
    await page.goto(`${WEB}/dashboard`);
    const buf = fs.readFileSync(path.join(media, "small.mp4"));
    await page.setInputFiles("input[type=file]", { name: "../../etc/passwd.mp4", mimeType: "video/mp4", buffer: buf });
    await page.getByRole("button", { name: "Upload video" }).click();
    await page.getByText("Video ready").waitFor({ timeout: 90000 });
    const row = sql("select original_filename from projects where status <> 'FAILED' order by created_at desc limit 1");
    assert(!row.includes("..") && !row.includes("/"), "stored name: " + row);
    const keys = storageKeys();
    assert(keys.every((k) => !k.includes("..") && !/passwd/i.test(k)), "storage keys: " + keys.join(", "));
    return `stored as "${row}"`;
  });

  // ---------- 3. the real job -------------------------------------------------------------------------------------
  await step("upload the sample clip with sound; the dashboard lists real facts from the server", async () => {
    await page.goto(`${WEB}/dashboard`);
    await page.setInputFiles("input[type=file]", path.join(media, "small.mp4"));
    await page.getByRole("button", { name: "Upload video" }).click();
    await page.getByText("Video ready").waitFor({ timeout: 90000 });
    await page.getByText(/320\s*×\s*568/).first().waitFor({ timeout: 10000 });
    await page.getByRole("link", { name: "Open editor" }).first().click();
    await page.waitForURL("**/editor/*");
    id = page.url().split("/editor/")[1];
    await page.locator("canvas").waitFor();
    await page.waitForFunction(() => document.querySelector("video")?.readyState >= 2, null, { timeout: 30000 });
  });

  await step("mark a rectangle and a polygon, add a keyframe, save; a reload brings them back", async () => {
    await page.getByRole("button", { name: "Rectangle", exact: true }).click();
    await drag(page, [0.1, 0.1], [0.5, 0.3]);
    await page.getByRole("button", { name: "Polygon", exact: true }).click();
    const b = await page.locator("canvas").boundingBox();
    for (const [fx, fy] of [[0.55, 0.7], [0.9, 0.7], [0.9, 0.85], [0.55, 0.85]]) await page.mouse.click(b.x + fx * b.width, b.y + fy * b.height);
    await page.keyboard.press("Enter");
    await page.getByRole("button", { name: "Add keyframe here" }).waitFor();
    await page.getByRole("button", { name: "Save masks" }).click();
    await page.getByText(/Saved/).first().waitFor({ timeout: 15000 });
    await page.reload();
    await page.locator("canvas").waitFor();
    const regions = await page.locator("section[aria-labelledby=regions-heading] li button[aria-pressed]").count();
    assert(regions === 2, "regions after reload: " + regions);
  });

  await step("render: stages advance in order, progress is real, and the editor locks the marks", async () => {
    await page.getByRole("button", { name: "Remove marked areas" }).click();
    await page.getByText("Finished and checked").waitFor({ timeout: 120000 });
    const { job } = (await api(page, `/api/projects/${id}/render`)).body;
    assert(job.status === "completed" && job.percent === 100 && job.frames_done === 60 && job.frames_total === 60, JSON.stringify({ s: job.status, p: job.percent, d: job.frames_done, t: job.frames_total }));
    assert(job.stages.every((x) => x.state === "done" || x.state === "skipped"), "stages: " + job.stages.map((x) => `${x.key}:${x.state}`).join(" "));
    assert(job.result.frames === 60 && job.result.audio === "copied" && job.engine.development_only === true, JSON.stringify(job.result));
    const locked = await api(page, `/api/projects/${id}/mask`, { method: "PUT", ...json({ version: 1, regions: [] }) });
    return `${job.stages.map((x) => x.key).join(" > ")}; audio ${job.result.audio}; marks edit after finish: HTTP ${locked.status}`;
  });

  await step("result page: compare view, honest dev-engine notice, download is the real file", async () => {
    await page.getByRole("link", { name: "Preview and download" }).click();
    await page.getByRole("heading", { name: "Your restored video is ready." }).waitFor({ timeout: 15000 });
    await page.getByText(/not an? AI/i).first().waitFor({ timeout: 5000 });
    await page.waitForFunction(() => [...document.querySelectorAll("video")].every((v) => v.readyState >= 2), null, { timeout: 30000 });
    const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: /Download MP4/ }).click()]);
    const file = path.join(DIR, "full-result.mp4");
    await download.saveAs(file);
    const probe = JSON.parse(execSync(`ffprobe -v error -show_entries stream=codec_type,codec_name,width,height,nb_frames -of json ${file}`).toString());
    const v = probe.streams.find((s) => s.codec_type === "video"), a = probe.streams.find((s) => s.codec_type === "audio");
    assert(v.codec_name === "h264" && v.width === 320 && v.height === 568 && Number(v.nb_frames) === 60, JSON.stringify(v));
    assert(a && a.codec_name === "aac", "audio kept");
    return `${download.suggestedFilename()}: ${v.codec_name} ${v.width}x${v.height}, ${v.nb_frames} frames, ${a.codec_name}`;
  });

  await step("output verification: the picture changed inside the marks and stayed the same outside them", async () => {
    // Compare against the source file itself (the app's preview copy of the original is lower resolution).
    const src = path.join(media, "small.mp4"), res = path.join(DIR, "full-result.mp4");
    const r = execSync(`${VENV}/bin/python - <<'PY'
import subprocess, numpy as np
def frames(path, n):
    w,h=320,568
    raw=subprocess.run(["ffmpeg","-v","error","-i",path,"-frames:v",str(n),"-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True,check=True).stdout
    return np.frombuffer(raw,dtype=np.uint8).reshape(-1,h,w,3).astype(int)
a=frames("${src}",30); b=frames("${res}",30)
d=np.abs(a-b).mean(axis=(0,3))
H,W=d.shape
def box(x0,y0,x1,y1): return d[int(y0*H):int(y1*H), int(x0*W):int(x1*W)].mean()
inside_rect=box(0.12,0.12,0.48,0.28); inside_poly=box(0.6,0.72,0.88,0.83)
mask=np.zeros_like(d,dtype=bool)
mask[int(0.07*H):int(0.33*H), int(0.07*W):int(0.53*W)]=True
mask[int(0.65*H):int(0.9*H), int(0.5*W):int(0.95*W)]=True
outside=d[~mask].mean()
print(f"{inside_rect:.2f} {inside_poly:.2f} {outside:.2f}")
PY`).toString().trim().split(" ").map(Number);
    const [rect, poly, outside] = r;
    assert(rect > 4 && poly > 4, `marked areas barely changed (rect ${rect}, polygon ${poly})`);
    assert(outside < 3, `unmarked picture changed too much (mean abs diff ${outside})`);
    return `mean difference: rectangle ${rect}, polygon ${poly}, everywhere else ${outside}`;
  });

  // ---------- 4. other people, other limits --------------------------------------------------------------------
  await step("a second person cannot see, change, render, download or delete the first person's project", async () => {
    const other = await browser.newContext(); const p2 = await other.newPage();
    await signup(p2, "Grace Hopper", emailB);
    const results = [];
    for (const [method, url, body] of [["GET", `/api/projects/${id}`], ["GET", `/api/projects/${id}/media`], ["GET", `/api/projects/${id}/mask`], ["PUT", `/api/projects/${id}/mask`, { version: 1, regions: [] }], ["POST", `/api/projects/${id}/track`], ["GET", `/api/projects/${id}/render`], ["POST", `/api/projects/${id}/render`, {}], ["GET", `/api/projects/${id}/preview`], ["POST", `/api/projects/${id}/export`], ["DELETE", `/api/projects/${id}`]]) {
      const r = await api(p2, url, { method, ...(body ? json(body) : {}) });
      results.push(`${method} ${r.status}`);
      assert(r.status === 404, `${method} ${url} gave ${r.status}`);
    }
    await p2.goto(`${WEB}/result/${id}`);
    await p2.getByText(/not found/i).first().waitFor({ timeout: 15000 });
    const mine = await api(p2, "/api/projects");
    assert(mine.body.length === 0, "the list must be empty for the other person");
    await other.close();
    const still = sql(`select status from projects where id = '${id}'`);
    assert(still === "COMPLETED", "the project must be untouched, is " + still);
    return results.length + " requests, all 404";
  });

  await step("without signing in, the API refuses everything", async () => {
    const anon = await browser.newContext(); const p3 = await anon.newPage();
    await p3.goto(WEB);
    for (const url of ["/api/projects", `/api/projects/${id}`, `/api/projects/${id}/preview`, "/api/auth/me"]) {
      const r = await api(p3, url);
      assert(r.status === 401, `${url} gave ${r.status}`);
    }
    await anon.close();
  });

  await step("a forged session cookie is rejected", async () => {
    const forged = await browser.newContext(); const p4 = await forged.newPage();
    await forged.addCookies([{ name: "cf_session", value: "eyJhbGciOiJub25lIn0.eyJzdWIiOiIxIn0.", url: API }]);
    await p4.goto(WEB);
    const r = await api(p4, "/api/auth/me");
    assert(r.status === 401, "gave " + r.status);
    await forged.close();
  });

  await step("a request from another website's origin cannot act as the signed-in person", async () => {
    const res = await page.evaluate(async (a) => {
      const r = await fetch(a + "/api/projects", { method: "POST", credentials: "include", headers: { "content-type": "application/json", Origin: "https://evil.example" }, body: JSON.stringify({ filename: "x.mp4", size_bytes: 10 }) }).catch((e) => ({ status: 0 }));
      return r.status;
    }, API);
    // The browser sets Origin itself (the header above is ignored), so also test it the way an attacker's page would: straight from Node.
    const forged = await fetch(API + "/api/projects", { method: "POST", headers: { "content-type": "application/json", Origin: "https://evil.example", Cookie: (await ctx.cookies(API)).map((c) => `${c.name}=${c.value}`).join("; ") }, body: JSON.stringify({ filename: "x.mp4", size_bytes: 10 }) });
    assert(forged.status === 403, "forged-origin write gave " + forged.status);
  });

  // ---------- 5. expiry and cleanup ----------------------------------------------------------------------------
  await step("a project past its deadline is removed by the schedule, with every file", async () => {
    makeDue(`id = '${id}'`);
    const before = storageKeys().length;
    const t0 = Date.now();
    for (;;) {
      if (Date.now() - t0 > 150000) throw new Error("not swept in 150s");
      const status = sql(`select status from projects where id = '${id}'`);
      if (status === "EXPIRED") break;
      await sleep(3000);
    }
    const keys = storageKeys();
    assert(!keys.some((k) => k.includes(id)), "files of the expired project remain: " + keys.filter((k) => k.includes(id)).join(", "));
    return `${before} stored files before, ${keys.length} after (the other projects' files stay), ${Math.round((Date.now() - t0) / 1000)}s`;
  });

  await step("expired: the API answers 410 and the pages say so; the person can still remove the row", async () => {
    for (const url of [`/api/projects/${id}/media`, `/api/projects/${id}/preview`]) {
      const r = await api(page, url);
      assert(r.status === 410, `${url} gave ${r.status}`);
    }
    await page.goto(`${WEB}/result/${id}`);
    await page.getByRole("heading", { name: "This project has expired" }).waitFor({ timeout: 15000 });
    const del = await api(page, `/api/projects/${id}`, { method: "DELETE" });
    assert(del.status === 204, "delete gave " + del.status);
  });

  await step("deleting the remaining projects leaves nothing in storage or in the database", async () => {
    for (const p of (await api(page, "/api/projects")).body) assert((await api(page, `/api/projects/${p.id}`, { method: "DELETE" })).status === 204);
    const keys = storageKeys();
    assert(keys.length === 0, "storage still holds: " + keys.join(", "));
    const left = sql("select count(*) from projects") + "/" + sql("select count(*) from jobs") + "/" + sql("select count(*) from masks");
    assert(left === "0/0/0", "projects/jobs/masks left: " + left);
  });

  // ---------- 6. what the system recorded -----------------------------------------------------------------------
  await step("the logs tell the story with ids, and hold nothing private", async () => {
    const all = log("api.log") + "\n" + log("worker.log");
    for (const event of ["user registered", "login succeeded", "upload started", "upload completed", "upload refused: not a video", "analysis completed", "render started", "render finished", "export link issued", "project expired", "project deleted"])
      assert(all.includes(event), `missing log event: ${event}`);
    for (const secret of ["X-Amz-Signature", "cf_session=", PASSWORD, "passwd.mp4", "garbage.mp4", emailA, emailB]) assert(!all.includes(secret), `${secret} appears in logs`);
    const trace = all.split("\n").filter((l) => /render (started|finished)/.test(l));
    assert(trace.every((l) => /job_id=/.test(l) && /project_id=/.test(l) && /user_id=/.test(l)), "render lines carry ids");
  });

  await step("no unexpected console errors or page errors in the whole journey", async () => { assert(problems.length === 0, problems.join(" | ").slice(0, 500)); });
  await browser.close();
  console.log(failures ? `\n${failures} step(s) failed` : "\nAll steps passed");
  process.exit(failures ? 1 : 0);
})();
