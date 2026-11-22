# MinIO with a shell, for hosts that start containers with `sh -c` (Railway). The official image has none, so the host
# can't start it with a command of its own. The binary is copied unchanged from the official image (pinned release).
# Build context: the repository root.   docker build -f docker/minio.Dockerfile -t cleanframe-minio .
FROM quay.io/minio/minio:RELEASE.2025-04-22T22-12-26Z AS minio

FROM debian:bookworm-slim
COPY --from=minio /usr/bin/minio /usr/bin/minio
RUN mkdir -p /data
EXPOSE 9000
CMD ["minio", "server", "/data", "--address", ":9000"]
