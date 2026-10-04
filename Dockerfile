# CineRec — Multi-Modal Movie Recommendation System
# Multi-stage Docker build. `APP_MODE=full` installs the training/inference stack
# (torch + vision), `lite` installs only the runtime deps needed for precomputed
# recommendations. Pass it as a build arg: `docker build --build-arg APP_MODE=lite .`
ARG APP_MODE=full

# Stage 1: build dependencies
FROM python:3.10-slim AS deps
ARG APP_MODE
WORKDIR /app
COPY requirements.txt requirements-train.txt ./
RUN if [ "$APP_MODE" = "full" ]; then \
        pip install --no-cache-dir -r requirements-train.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Stage 2: runtime
FROM python:3.10-slim
WORKDIR /app

# curl is used by the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed Python packages from the build stage
COPY --from=deps /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Create an unprivileged user and copy the application owned by it
RUN useradd --create-home --uid 10001 appuser
COPY . .
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]