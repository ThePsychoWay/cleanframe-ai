# Browser journeys

Real-browser tests against the real API, worker, scheduler, Redis, an S3 stand-in and the database in `DATABASE_URL`
(SQLite by default, PostgreSQL if you give it an address). Chromium only.

1. Python environment with `backend/requirements-dev.txt` installed; `export CF_VENV=<its path>`
2. `npm i -g playwright && npx playwright install chromium`
3. Build the web app with the storage address the policy must allow:
   `cd frontend && NEXT_PUBLIC_STORAGE_ORIGIN=http://localhost:5000 npm run build`
4. `scripts/e2e/make_media.sh` (two small input files), then `scripts/e2e/start.sh`
5. `scripts/e2e/run.sh e2e_full.js` (19 steps: sign-in, bad uploads, marks, render, download and pixel check, other
   accounts, forged cookies and origins, expiry, cleanup, logs) and `scripts/e2e/run.sh e2e_security.js` (headers, policy,
   lockout, scheduled deletion)
6. `scripts/e2e/stop.sh`

`e2e_full.js` looks up rows with PostgreSQL-style ids, so run it with `DATABASE_URL` pointing at PostgreSQL: on SQLite four
steps fail only because those lookups find nothing (the app itself behaves the same). `e2e_removal.js` (Phase 15: the
`auto` engine through the real worker, result page and download, a fading logo with the `measuring` stage observed while it runs, plus the "cannot be rebuilt" failure shown to the person)
runs on either: `scripts/e2e/make_removal_media.sh`, start the stack with `AI_PROVIDER=auto`, `run.sh e2e_removal.js`.

`start.sh` wipes the database it points at (the PostgreSQL `public` schema, or the SQLite file in `CF_E2E_DIR`).
Nothing here is a real credential; the secrets are random or fixed placeholders for the local stand-ins.
