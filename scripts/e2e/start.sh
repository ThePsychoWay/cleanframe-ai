#!/bin/bash
# Starts Redis, a local S3 (moto), the API, a worker, the scheduler and the built web app on fresh data.
# Build the web app first: (cd frontend && NEXT_PUBLIC_STORAGE_ORIGIN=http://localhost:5000 npm run build)
source "$(dirname "$0")/env.sh"
L="$CF_E2E_DIR"
bash "$(dirname "$0")/stop.sh"
rm -f "$L"/*.log "$L"/app.db "$L"/celerybeat-schedule*
case "$DATABASE_URL" in
  postgresql*) "$CF_VENV/bin/python" -c "
import os; from sqlalchemy import create_engine, text
e = create_engine(os.environ['DATABASE_URL'])
with e.begin() as c: c.execute(text('DROP SCHEMA public CASCADE')); c.execute(text('CREATE SCHEMA public'))" ;;
esac
redis-server --port 6379 --save "" --daemonize yes --logfile "$L/redis.log" >/dev/null
cd "$CF_ROOT/backend"
nohup "$CF_VENV/bin/moto_server" -p 5000 > "$L/moto.log" 2>&1 &
for _ in $(seq 20); do curl -s -o /dev/null localhost:5000 && break; sleep 0.5; done
"$CF_VENV/bin/python" - <<PY
import boto3
s3 = boto3.client("s3", endpoint_url="http://localhost:5000", region_name="us-east-1")
s3.create_bucket(Bucket="cleanframe")
s3.put_bucket_cors(Bucket="cleanframe", CORSConfiguration={"CORSRules": [{"AllowedOrigins": ["http://localhost:3000"], "AllowedMethods": ["PUT", "GET", "HEAD"], "AllowedHeaders": ["*"], "ExposeHeaders": ["ETag"]}]})
PY
"$CF_VENV/bin/alembic" upgrade head > "$L/alembic.log" 2>&1 || { echo "migrations failed:"; tail -5 "$L/alembic.log"; exit 1; }
nohup "$CF_VENV/bin/uvicorn" app.main:app --port 8000 > "$L/api.log" 2>&1 &
nohup "$CF_VENV/bin/celery" -A app.workers.celery_app worker -Q cpu --concurrency="${WORKER_CONCURRENCY:-1}" > "$L/worker.log" 2>&1 &
nohup "$CF_VENV/bin/celery" -A app.workers.celery_app beat --schedule="$L/celerybeat-schedule" > "$L/beat.log" 2>&1 &
cd "$CF_ROOT/frontend"
nohup npx next start -p 3000 > "$L/web.log" 2>&1 &
for _ in $(seq 40); do curl -s -o /dev/null localhost:3000 && curl -s localhost:8000/api/health | grep -q ok && break; sleep 0.5; done
curl -s localhost:8000/api/health; echo; curl -s -o /dev/null -w "web %{http_code}\n" localhost:3000
