"""Application settings, loaded from environment variables. No secrets are hardcoded."""

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIN_JWT_SECRET_LENGTH = 32


class ConfigError(RuntimeError):
    """Raised at startup when required configuration is missing or unsafe."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Local mode: everything on one computer with nothing to install or sign up for (no PostgreSQL, Redis, S3 or API
    # keys): a SQLite file, the disk instead of object storage, and background jobs on threads in the API process.
    # It only fills in what you left unset, so any of these can still be overridden. See docs/LOCAL.md.
    local_mode: bool = False
    local_data_dir: str = "cleanframe-data"
    # Address the browser uses to reach this API; signed file links for the disk storage point here. Empty = relative links
    # (the page's own address), which is what phone access (`run_local.py --lan`) uses: everything goes through the web app's port.
    api_public_url: str = "http://localhost:8000"
    # Local mode reached from other devices on the home network (a phone). Only `run_local.py --lan` sets it; see docs/LOCAL.md.
    local_lan: bool = False
    storage_backend: Literal["s3", "filesystem"] = "s3"
    task_backend: Literal["celery", "thread"] = "celery"
    # Jobs that run at the same time in local mode. 1 keeps memory low: renders and analysis take turns.
    local_workers: int = Field(default=1, ge=1, le=4)
    # Memory the overlay measurement may use (the more it has, the more frames it studies). Fewer than 16 frames are
    # never used, so a very large marked area can need more than this.
    overlay_memory_mb: int = Field(default=1024, ge=256)
    # Memory the frame-search window of the reconstruction may use. A 4K frame is 25 MB, so a long window is cut to fit.
    temporal_memory_mb: int = Field(default=1536, ge=256)

    # Core services
    database_url: str = ""
    redis_url: str = "redis://localhost:6379/0"

    # Object storage. Empty endpoint means AWS S3; set it for MinIO or another S3-compatible store.
    storage_endpoint: str = ""
    # Address browsers use for presigned uploads when it differs from the one the server uses
    # (for example http://localhost:9000 in the browser vs http://minio:9000 inside Docker).
    storage_public_endpoint: str = ""
    storage_access_key: str = ""
    storage_secret_key: str = ""
    storage_bucket: str = "cleanframe"
    storage_region: str = "us-east-1"
    # How bucket names appear in URLs: path (host/bucket/key; MinIO) or virtual (bucket.host/key; Railway buckets, AWS S3).
    storage_addressing_style: Literal["path", "virtual"] = "path"
    # Backstop for files nothing else removed (a crashed worker, an abandoned upload): the bucket expires them this many
    # days after the retention period ends. Applied by `python -m app.storage.provision`.
    storage_lifecycle_margin_days: int = Field(default=2, ge=1)
    # Incomplete multipart uploads are discarded by the bucket this many days after they were started.
    storage_abort_uploads_days: int = Field(default=2, ge=1)
    # S3 requires at least 5 MiB per part (except the last) and at most 10,000 parts.
    upload_part_size: int = Field(default=8 * 1024 * 1024, ge=5 * 1024 * 1024)

    # Auth
    jwt_secret: str = ""
    jwt_expire_minutes: int = Field(default=720, gt=0)
    session_cookie_name: str = "cf_session"
    # Comma-separated browser origins allowed to call the API with credentials.
    cors_origins: str = "http://localhost:3000"

    # AI
    ai_provider: Literal["dev", "auto", "overlay", "reconstruct", "lama", "local", "video", "api"] = "dev"
    ai_api_key: str = ""
    ai_api_url: str = ""
    ai_api_timeout_seconds: float = Field(default=60.0, gt=0)
    # TorchScript file for AI_PROVIDER=local (one image at a time) or video (a clip at a time).
    ai_model_path: str = ""
    # Optional: where to download the model file from when AI_MODEL_PATH doesn't exist yet. HTTPS only, and the file is
    # checked against AI_MODEL_SHA256 (required with the URL) before it is used.
    ai_model_url: str = ""
    ai_model_sha256: str = ""
    # AI_PROVIDER=reconstruct (and auto, which tries overlay removal first) rebuilds marked areas from other frames. What no frame shows (a mark that never moves on a
    # camera that never moves) goes to LaMa ("lama", needs AI_MODEL_PATH to an ONNX LaMa model), or the render stops
    # with a message ("none"). "auto" = lama when a LaMa model is configured, otherwise none.
    reconstruct_fallback: Literal["auto", "lama", "none"] = "auto"
    ai_video_clip_frames: int = Field(default=16, ge=3)
    ai_video_overlap_frames: int = Field(default=4, ge=0)
    cuda_enabled: Literal["auto", "true", "false"] = "auto"
    # Video encoder: auto = NVIDIA NVENC when a test encode proves it works here, otherwise the CPU; cpu = never NVENC.
    video_encoder: Literal["auto", "cpu", "nvenc"] = "auto"
    # Fully automatic mode: after an upload is analysed, find the overlays and start the render without anyone marking
    # anything. AUTO_APPEARANCE lets a small mark in a corner of a part of the picture that does not move be judged by how it
    # looks; with nothing moving around it there is no proof, so the result says "judged by look only". Turn it off to
    # remove only what motion proves.
    auto_process: bool = True
    auto_appearance: bool = True
    # Keep the generative model loaded between videos (and load it when the app starts) instead of loading it for every video.
    # Costs the model's memory (~0.2 GB) while idle; set false to give it back after each video.
    ai_keep_model_loaded: bool = True

    # Malware scanning of every upload before it is analysed. "clamav" sends the file to a clamd server (see
    # docker-compose, profile "scan"); clamd's StreamMaxLength must be at least MAX_UPLOAD_SIZE. If scanning is on and
    # the scanner can't be reached, uploads are NOT processed (it fails closed).
    malware_scan: Literal["off", "clamav"] = "off"
    clamav_host: str = ""
    clamav_port: int = Field(default=3310, ge=1, le=65535)
    clamav_timeout_seconds: int = Field(default=300, gt=0)

    # Limits and retention
    # The size of one upload and the most pixels per frame are limits of this server, not of the format: a 4K clip of a
    # minute is a few hundred MB. Raise MAX_UPLOAD_SIZE (bytes) for longer or higher-bitrate footage.
    max_upload_size: int = Field(default=2 * 1024 * 1024 * 1024, gt=0)
    max_video_duration_seconds: int = Field(default=120, gt=0)
    video_retention_hours: int = Field(default=48, gt=0)
    # How often the retention sweep runs (Celery beat). Access stops at the deadline regardless; this only decides how
    # soon after it the files are actually removed.
    cleanup_interval_seconds: int = Field(default=900, ge=60)
    max_video_pixels: int = Field(default=4096 * 2304, gt=0)  # 4096x2304 is the DCI-4K / 4K-UHD ceiling
    # Free disk space that must remain after a job's temporary files and result are accounted for.
    min_free_disk_mb: int = Field(default=512, ge=0)
    # API request bodies are small JSON (video goes straight to storage), so this is a hard cap.
    max_request_body_bytes: int = Field(default=2 * 1024 * 1024, gt=0)

    # Tracking can run for minutes on long videos. The worker stops it at the soft limit; a job that is
    # still marked running well past the hard limit is treated as lost (the worker died) and reported failed.
    tracking_time_limit_seconds: int = Field(default=1500, gt=0)

    # Background jobs. Eager mode runs tasks inline (tests and single-process development only).
    celery_eager: bool = False

    # Rendering (the long job). The worker stops at the soft limit; the hard limit is 5 minutes later.
    render_time_limit_seconds: int = Field(default=7200, gt=0)
    # When the after-render check finds the mark still showing or the fill flickering, render once more at the highest
    # quality and keep whichever of the two scored better. Only while less than half the render time limit is used.
    qc_retry: bool = True
    # Which queue renders go to: "auto" sends engines that need a GPU (local, video) to `gpu` and the rest to `cpu`.
    # A machine without a GPU can run one worker for both: `celery ... worker -Q cpu,gpu`.
    render_queue: Literal["auto", "cpu", "gpu"] = "auto"
    max_active_jobs_per_user: int = Field(default=2, ge=1)
    # A worker writes a heartbeat this often; a running job with none for `job_lost_after_seconds` is reported failed.
    job_heartbeat_seconds: float = Field(default=15.0, gt=0)
    job_lost_after_seconds: float = Field(default=120.0, gt=0)
    # A job nobody picked up for this long (no worker running, or the queue was wiped) is reported failed.
    job_queue_timeout_seconds: int = Field(default=7200, gt=0)
    # Where workers keep the original, the rebuilt picture and the result while rendering. Empty = system temp.
    worker_tmp_dir: str = ""

    # Rate limiting (app/ratelimit.py). "auto" uses Redis unless DEV_MODE is on; "memory" is refused in production
    # because each API process would count on its own.
    rate_limit_enabled: bool = True
    rate_limit_storage: Literal["auto", "memory", "redis"] = "auto"
    # How many reverse proxies sit in front of the API and append to X-Forwarded-For. 0 = none: the header is ignored.
    trusted_proxy_count: int = Field(default=0, ge=0, le=5)

    # Logging: "auto" = JSON lines in production, readable text when DEV_MODE is on.
    log_level: str = "INFO"
    log_format: Literal["auto", "json", "text"] = "auto"

    # Development
    dev_mode: bool = True

    @model_validator(mode="after")
    def _fill_in_local_mode(self) -> "Settings":
        if not self.local_mode:
            return self
        data = Path(self.local_data_dir).expanduser().resolve()
        data.mkdir(parents=True, exist_ok=True)
        if not self.database_url:
            self.database_url = f"sqlite+pysqlite:///{data / 'cleanframe.db'}"
        if "storage_backend" not in self.model_fields_set:
            self.storage_backend = "filesystem"
        if "task_backend" not in self.model_fields_set:
            self.task_backend = "thread"
        if "ai_provider" not in self.model_fields_set:
            self.ai_provider = "auto"  # real removal (overlay inversion, then rebuilding from other frames); "dev" only blurs
        if "rate_limit_storage" not in self.model_fields_set:
            self.rate_limit_storage = "memory"
        if not self.worker_tmp_dir:
            self.worker_tmp_dir = str(data / "tmp")
            Path(self.worker_tmp_dir).mkdir(parents=True, exist_ok=True)
        if not self.jwt_secret:
            self.jwt_secret = _local_secret(data)
        return self

    @property
    def log_json(self) -> bool:
        return self.log_format == "json" or (self.log_format == "auto" and not self.dev_mode)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.cors_origins.split(",") if o.strip()]

    @property
    def cookie_secure(self) -> bool:
        """Session cookies are Secure everywhere except local development over http."""
        return not self.dev_mode and not self.local_mode  # local mode is plain http on localhost

    def validate_runtime(self) -> None:
        """Fail fast with an actionable message instead of failing on the first request."""
        if not self.database_url:
            raise ConfigError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
        if len(self.jwt_secret) < MIN_JWT_SECRET_LENGTH:
            raise ConfigError(
                f"JWT_SECRET must be at least {MIN_JWT_SECRET_LENGTH} characters. Generate one with: "
                'python -c "import secrets; print(secrets.token_urlsafe(64))"'
            )
        if not self.cors_origin_list:
            raise ConfigError("CORS_ORIGINS must list at least one allowed origin.")
        if not self.storage_bucket:
            raise ConfigError("STORAGE_BUCKET is not set.")
        if self.local_lan and not self.local_mode:
            raise ConfigError("LOCAL_LAN only applies to local mode (LOCAL_MODE=true).")
        if self.local_lan and self.api_public_url.startswith(("http://", "https://")) is False and self.api_public_url != "":
            raise ConfigError("With LOCAL_LAN, API_PUBLIC_URL must be empty (relative links) or a full http address.")
        if self.local_mode and not self.local_lan and self.cors_origin_list and not self.api_public_url.startswith(("http://localhost", "http://127.0.0.1")):
            raise ConfigError("Local mode serves files over plain http and is meant for this computer only (API_PUBLIC_URL must be localhost).")
        if not self.dev_mode and not self.rate_limit_enabled:
            raise ConfigError("RATE_LIMIT_ENABLED=false is for development only. Rate limiting can't be turned off in production.")
        if not self.dev_mode and self.rate_limit_storage == "memory":
            raise ConfigError(
                "RATE_LIMIT_STORAGE=memory is for development only: each API process would count separately. "
                "Use redis (or auto)."
            )
        if not self.dev_mode and self.ai_provider == "dev":
            raise ConfigError(
                "AI_PROVIDER=dev is the development-only engine and is refused when DEV_MODE is off. "
                "Set AI_PROVIDER to auto, overlay, reconstruct, lama, local, video or api."
            )
        if self.malware_scan == "clamav" and not self.clamav_host:
            raise ConfigError("MALWARE_SCAN=clamav needs CLAMAV_HOST (the clamd server's address).")
        if self.ai_model_url or self.ai_model_sha256:
            from app.ai.weights import config_problem

            problem = config_problem(self.ai_model_url, self.ai_model_sha256, allow_insecure_loopback=self.dev_mode)
            if problem:
                raise ConfigError(problem)
            if self.ai_model_url and not self.ai_model_path:
                raise ConfigError("AI_MODEL_URL needs AI_MODEL_PATH: where to save the downloaded file.")
        if self.ai_provider in ("auto", "reconstruct") and self.reconstruct_fallback == "lama" and not self.ai_model_path:
            raise ConfigError("RECONSTRUCT_FALLBACK=lama needs AI_MODEL_PATH (an ONNX LaMa model).")


def _local_secret(data: Path) -> str:
    """The session-signing secret for local mode: made once, kept in the data folder, readable only by this user."""
    path = data / "secret.key"
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if len(value) >= MIN_JWT_SECRET_LENGTH:
            return value
    value = secrets.token_urlsafe(64)
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)
    return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
