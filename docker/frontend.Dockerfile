# Web app image. Build context is the repository root:
#   docker build -f docker/frontend.Dockerfile -t cleanframe-web \
#     --build-arg NEXT_PUBLIC_API_URL=https://api.example.com \
#     --build-arg NEXT_PUBLIC_STORAGE_ORIGIN=https://files.example.com .
# NEXT_PUBLIC_* values are compiled into the page and into its security policy, so they are build arguments, not
# runtime settings: build one image per address the browser will use.
FROM node:22-slim AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

FROM node:22-slim AS build
WORKDIR /app
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ARG NEXT_PUBLIC_STORAGE_ORIGIN=
ARG NEXT_PUBLIC_SITE_URL=http://localhost:3000
ARG NEXT_PUBLIC_MAX_UPLOAD_SIZE=
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_PUBLIC_STORAGE_ORIGIN=$NEXT_PUBLIC_STORAGE_ORIGIN \
    NEXT_PUBLIC_SITE_URL=$NEXT_PUBLIC_SITE_URL \
    NEXT_PUBLIC_MAX_UPLOAD_SIZE=$NEXT_PUBLIC_MAX_UPLOAD_SIZE \
    NEXT_OUTPUT=standalone \
    NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ ./
RUN npm run build

FROM node:22-slim AS run
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
# Only what the server needs: the traced server files and the static assets. Runs as the image's ordinary user.
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
USER node
EXPOSE 3000
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD node -e "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.status<500?0:1)).catch(()=>process.exit(1))"
CMD ["node", "server.js"]
