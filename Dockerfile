# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    ANYDOCS_DATA_DIR=/data

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so a code change does not re-resolve the environment.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev --extra crawl

COPY anydocs/ ./anydocs/
COPY cli/ ./cli/
COPY config/ ./config/
COPY analytics/ ./analytics/
# The container is the learning host, so it installs the crawl extra the Vercel function omits.
RUN uv sync --frozen --no-dev --extra crawl

# Snapshots live outside the image: a corpus is data, not a build artifact.
VOLUME ["/data"]
EXPOSE 8000

# Readiness waits for the default snapshots to load, so this is not merely a liveness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

ENTRYPOINT ["uv", "run", "--no-dev", "anydocs"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
