# Environment for the local end-to-end stack. Sourced by start.sh and by the test scripts.
# Needs: CF_VENV (a Python environment with backend/requirements-dev.txt installed), ffmpeg, redis-server, Node 20+.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CF_ROOT="$ROOT"
export CF_E2E_DIR="${CF_E2E_DIR:-/tmp/cleanframe-e2e}"
export CF_VENV="${CF_VENV:-$ROOT/backend/.venv}"
mkdir -p "$CF_E2E_DIR/media"

# SQLite by default; set DATABASE_URL to a PostgreSQL address (postgresql+psycopg://user@host:port/db) to run on that.
export DATABASE_URL="${DATABASE_URL:-sqlite+pysqlite:///$CF_E2E_DIR/app.db}"
# Fresh random secret for each shell; nothing here is a real credential.
export JWT_SECRET="${JWT_SECRET:-$(openssl rand -hex 32)}"
export DEV_MODE=true
export CORS_ORIGINS=http://localhost:3000
export REDIS_URL=redis://localhost:6379/0
export STORAGE_ENDPOINT=http://localhost:5000
export STORAGE_PUBLIC_ENDPOINT=http://localhost:5000
export STORAGE_ACCESS_KEY=testing
export STORAGE_SECRET_KEY=testing-not-a-secret
export STORAGE_BUCKET=cleanframe
export UPLOAD_PART_SIZE=5242880
export CELERY_EAGER=false
export JOB_HEARTBEAT_SECONDS=2
export JOB_LOST_AFTER_SECONDS=12
export AI_PROVIDER="${AI_PROVIDER:-dev}"
export CLEANUP_INTERVAL_SECONDS=60
export RATE_LIMIT_ENABLED="${RATE_LIMIT_ENABLED:-true}"
export RATE_LIMIT_STORAGE=memory
# Read when the frontend is BUILT: lets the production policy allow the local storage address.
export NEXT_PUBLIC_STORAGE_ORIGIN=http://localhost:5000
