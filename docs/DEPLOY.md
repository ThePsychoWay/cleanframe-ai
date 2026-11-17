# Putting CleanFrame AI online (Vercel website + one backend server)

Only remove overlays from content you own or have permission to edit.

Vercel runs the **website** only. Removing a watermark needs FFmpeg, a disk, jobs that run for minutes and 2-8 GB of
memory, which Vercel's functions cannot provide. So the backend (API, worker, database, Redis, file storage) runs on one
server you rent, and the website on Vercel talks to it.

```
browser ──► your-site.vercel.app ──/api/*──► (Vercel forwards) ──► https://api.example.com  (Caddy ► API)
        └─► uploads/downloads of video  ────────────────────────► https://files.example.com (Caddy ► MinIO)
```

Why `/api/*` is forwarded: the login cookie is `SameSite=Lax`. If the browser called the API's own domain directly, it
would not send the cookie from your Vercel site. Forwarding makes the API look like part of the website.

## 1. The server (once)

- A Linux server with **Docker**, **at least 8 GB RAM** and 40 GB+ disk (the worker is limited to 8 GB by default; a smaller
  server needs `WORKER_MEMORY_LIMIT=4g` and shorter or smaller videos). CPU-only: a 5-second 720p clip with the fill
  model took about 3 minutes on 2 cores.
- Two DNS A records pointing at its public IP: `api.<your domain>` and `files.<your domain>`.
- Ports 80 and 443 open.

## 2. Settings (`.env` on the server, copied from `.env.example`)

Set: `POSTGRES_PASSWORD`, `STORAGE_ACCESS_KEY`, `STORAGE_SECRET_KEY`, `JWT_SECRET`, `DEV_MODE=false`, `AI_PROVIDER=auto`,
`API_DOMAIN`, `STORAGE_DOMAIN`, `STORAGE_PUBLIC_ENDPOINT=https://<STORAGE_DOMAIN>`,
`CORS_ORIGINS=https://<your-project>.vercel.app` (add a custom domain here too if you use one),
`NEXT_PUBLIC_STORAGE_ORIGIN=https://<STORAGE_DOMAIN>`.

## 3. Start the backend

```
docker compose --env-file .env -f docker/docker-compose.yml -f docker/docker-compose.prod.yml up -d --build api worker beat postgres redis minio caddy
```

Check `https://api.<your domain>/api/health` answers. Only Caddy is open to the internet.

## 4. The Vercel project

- Import the repository, **Root Directory = `frontend`**, framework Next.js.
- Environment variables (Production and Preview):
  - `NEXT_PUBLIC_API_URL` = `same-origin`
  - `API_PROXY_TARGET` = `https://api.<your domain>`
  - `NEXT_PUBLIC_STORAGE_ORIGIN` = `https://files.<your domain>`
  - `NEXT_PUBLIC_SITE_URL` = your site's address
- Deploy. Open the site, create an account, upload a short clip.

`API_PROXY_TARGET` is read when the site is built, so after changing it redeploy.

## Known limits (not tested here)

- **Nothing in this guide was run on a real server or on Vercel.** The compose file, the Caddy file and the forwarding
  were checked for validity and the forwarding was checked against a stand-in API on one machine.
- The Docker setup (PostgreSQL, Redis, MinIO, Celery) was prepared in earlier phases but never built end to end where I
  could pull images; expect to fix small things on the first run.
- The live progress stream (`/render/events`) goes through Vercel's forwarding, which may cut long connections; if it does,
  the page still shows the result when the job finishes, but live progress may lag.
- Vercel's free Hobby plan is for non-commercial use; a business needs the Pro plan.
- Single-user local mode (`run_local.py`) is not for hosting: it refuses non-localhost addresses by design.
