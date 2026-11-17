const { chromium } = require("playwright");
const path = require("path");
const { media, sql, makeDue, storageKeys, log } = require("./lib");
const WEB = "http://localhost:3000", API = "http://localhost:8000";
let failures = 0;
async function step(name, fn) {
  try { const d = await fn(); console.log(`PASS  ${name}${d ? "  — " + d : ""}`); }
  catch (e) { failures += 1; console.log(`FAIL  ${name}  — ${String(e.message).split("\n")[0]}`); }
}
const assert = (c, m) => { if (!c) throw new Error(m || "assertion failed"); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function signup(page, name, email) {
  await page.goto(`${WEB}/signup`);
  await page.fill("#name", name); await page.fill("#email", email); await page.fill("#password", "correct horse battery");
  await page.click("button[type=submit]");
  await page.waitForURL("**/dashboard", { timeout: 15000 });
}
const fetchJson = (page, p, opts) => page.evaluate(async ([api, p, opts]) => { const r = await fetch(api + p, { credentials: "include", ...(opts || {}) }); return { status: r.status, body: r.status === 204 ? null : await r.json().catch(() => null) }; }, [API, p, opts]);

(async () => {
  const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 1100 } });
  const page = await ctx.newPage();
  const problems = [];
  page.on("pageerror", (e) => problems.push("pageerror: " + e.message));
  page.on("console", (m) => { if (m.type() === "error" && !/status of (401|404|410|429)/.test(m.text())) problems.push("console: " + m.text()); });
  const email = `sec${Date.now()}@example.com`;
  let id;

  await step("security headers come with the page and the API", async () => {
    const res = await page.goto(`${WEB}/login`);
    const h = res.headers();
    assert(/default-src 'self'/.test(h["content-security-policy"] || ""), "page CSP missing");
    assert(/frame-ancestors 'none'/.test(h["content-security-policy"]) && /object-src 'none'/.test(h["content-security-policy"]), "CSP directives");
    assert(h["x-frame-options"] === "DENY" && h["x-content-type-options"] === "nosniff", "page headers");
    assert(/camera=\(\)/.test(h["permissions-policy"] || ""), "permissions policy");
    assert(!h["x-powered-by"], "x-powered-by should be gone");
    const api = Object.fromEntries((await fetch(API + "/api/health")).headers.entries());
    assert(api["x-request-id"] && api["x-content-type-options"] === "nosniff" && api["x-frame-options"] === "DENY" && api["cache-control"] === "no-store", JSON.stringify(api));
    return `request id ${api["x-request-id"]}`;
  });

  await step("with the policy on: sign up, upload, mark, render, and play the result with nothing blocked", async () => {
    await signup(page, "Ada Lovelace", email);
    await page.setInputFiles("input[type=file]", path.join(media, "small.mp4"));
    await page.getByRole("button", { name: "Upload video" }).click();
    await page.getByText("Video ready").waitFor({ timeout: 90000 });
    await page.getByRole("link", { name: "Open editor" }).first().click();
    await page.waitForURL("**/editor/*");
    id = page.url().split("/editor/")[1];
    await page.locator("canvas").waitFor();
    await page.waitForFunction(() => document.querySelector("video")?.readyState >= 2, null, { timeout: 30000 });
    const b = await page.locator("canvas").boundingBox();
    await page.getByRole("button", { name: "Rectangle", exact: true }).click();
    await page.mouse.move(b.x + b.width * 0.1, b.y + b.height * 0.1); await page.mouse.down();
    await page.mouse.move(b.x + b.width * 0.5, b.y + b.height * 0.3, { steps: 8 }); await page.mouse.up();
    await page.getByRole("button", { name: "Save masks" }).click();
    await page.getByText(/Saved/).first().waitFor({ timeout: 15000 });
    await page.getByRole("button", { name: "Remove marked areas" }).click();
    await page.getByText("Finished and checked").waitFor({ timeout: 120000 });
    await page.getByRole("link", { name: "Preview and download" }).click();
    await page.getByRole("heading", { name: "Your restored video is ready." }).waitFor({ timeout: 15000 });
    await page.waitForFunction(() => [...document.querySelectorAll("video")].every((v) => v.readyState >= 2), null, { timeout: 30000 });
    const blocked = problems.filter((p) => /Content Security Policy|Refused to/i.test(p));
    assert(blocked.length === 0, "CSP blocked something: " + blocked.join(" | ").slice(0, 300));
  });

  await step("the dashboard says when the files will be deleted", async () => {
    await page.goto(`${WEB}/dashboard`);
    await page.getByText(/Files are deleted automatically on/).first().waitFor({ timeout: 15000 });
    const text = await page.locator("#recent-projects").innerText();
    assert(/Files are deleted automatically on /.test(text), "retention line missing: " + text.slice(0, 200));
    return (text.match(/Files are deleted automatically on [^.]*\d{4}[^.]*/) || [""])[0];
  });

  await step("repeated wrong passwords lock the account and the page says so in plain words", async () => {
    const other = await browser.newContext(); const p2 = await other.newPage();
    const victim = `lock${Date.now()}@example.com`;
    await signup(p2, "Grace Hopper", victim);
    await p2.request.post(`${API}/api/auth/logout`).catch(() => undefined);
    await p2.context().clearCookies();
    let message = "";
    for (let i = 0; i < 9; i++) {
      await p2.goto(`${WEB}/login`);
      await p2.fill("#email", victim); await p2.fill("#password", i === 8 ? "correct horse battery" : "wrong password " + i);
      await p2.click("button[type=submit]");
      await p2.locator("form [role=alert]").waitFor({ timeout: 15000 });
      message = await p2.locator("form [role=alert]").innerText();
    }
    assert(/Too many failed sign-in attempts for this account/.test(message) && /minutes/.test(message), "message: " + message);
    assert(p2.url().includes("/login"), "the correct password must not get in while locked");
    await other.close();
    return message;
  });

  await step("a project past its deadline is deleted by the scheduled sweep, not by anyone opening it", async () => {
    const before = storageKeys();
    assert(before.length >= 4, "expected stored files before: " + before);
    makeDue();
    const t0 = Date.now();
    for (;;) {
      if (Date.now() - t0 > 150000) throw new Error("the sweep never ran; beat log: " + log("beat.log").slice(-300));
      const keys = storageKeys();
      if (keys.length === 0) break;
      await sleep(3000);
    }
    const status = sql("select status from projects");
    assert(status === "EXPIRED", "status " + status);
    return `all files gone after ${Math.round((Date.now() - t0) / 1000)}s, status ${status}`;
  });

  await step("the dashboard, the result page and the editor all say the project expired", async () => {
    await page.goto(`${WEB}/dashboard`);
    await page.getByText(/files were deleted/).first().waitFor({ timeout: 15000 });
    const row = await page.locator("#recent-projects").innerText();
    assert(/Expired/.test(row) && /files were deleted/.test(row), "dashboard: " + row.slice(0, 200));
    assert(!(await page.getByRole("link", { name: /Open result|Open editor|Edit marks/ }).count()), "no open links for an expired project");
    await page.goto(`${WEB}/result/${id}`);
    await page.getByRole("heading", { name: "This project has expired" }).waitFor({ timeout: 15000 });
    await page.goto(`${WEB}/editor/${id}`);
    await page.getByRole("heading", { name: "This project has expired" }).waitFor({ timeout: 15000 });
    const r = await fetchJson(page, `/api/projects/${id}/preview`);
    assert(r.status === 410 && /expired/.test(r.body.detail), JSON.stringify(r));
  });

  await step("an expired project can still be removed from the list", async () => {
    await page.goto(`${WEB}/dashboard`);
    await page.getByRole("button", { name: /^Delete / }).first().click();
    await page.getByRole("button", { name: "Delete", exact: true }).click();
    await page.getByText("No projects yet").waitFor({ timeout: 15000 });
  });

  await step("logs: structured, every request has an id, and nothing private is in them", async () => {
    const api = log("api.log");
    const worker = log("worker.log");
    const all = api + "\n" + worker;
    assert(/request_id=[0-9a-f]{16}/.test(api), "access lines with request ids");
    assert(/retention sweep finished.*expired=1/.test(worker), "the sweep logged its result");
    assert(/render finished.*duration_ms=\d+/.test(worker) && /user_id=/.test(worker.split("\n").filter((l) => /render finished/.test(l))[0] || ""), "render finished with ids and duration");
    for (const secret of ["X-Amz-Signature", "X-Amz-Credential", "cf_session=", "small.mp4", "correct horse battery", "wrong password"]) assert(!all.includes(secret), `${secret} found in logs`);
    return `${api.split("\n").length} api lines, ${worker.split("\n").length} worker lines scanned`;
  });

  await step("no unexpected console errors anywhere in the run", async () => { assert(problems.length === 0, problems.join(" | ").slice(0, 400)); });
  await browser.close();
  console.log(failures ? `\n${failures} step(s) failed` : "\nAll steps passed");
  process.exit(failures ? 1 : 0);
})();
