#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04}"
NODE_BASE_IMAGE="${NODE_BASE_IMAGE:-node:20-alpine}"
PULL_RETRY_COUNT="${PULL_RETRY_COUNT:-5}"
PULL_RETRY_DELAY="${PULL_RETRY_DELAY:-15}"

log() {
  echo "[install] $*"
}

fail() {
  echo "[install][error] $*" >&2
  exit 1
}

require_cmd() {
  local name="$1"
  command -v "$name" >/dev/null 2>&1 || fail "Missing required command: $name"
}

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
    return
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
    return
  fi
  fail "Docker Compose not found. Install docker compose v2 or docker-compose."
}

pull_with_retry() {
  local image="$1"
  local attempts="$2"
  local delay="$3"
  local try=1

  while true; do
    if docker pull "$image"; then
      return 0
    fi
    if (( try >= attempts )); then
      fail "Failed to pull $image after $attempts attempts"
    fi
    log "Pull failed for $image (attempt $try/$attempts). Retrying in ${delay}s"
    sleep "$delay"
    ((try+=1))
  done
}

main() {
  cd "$ROOT_DIR"

  require_cmd docker
  require_cmd npm
  require_cmd python3

  detect_compose

  log "Installing web dependencies"
  if [[ -f web/package-lock.json ]]; then
    (cd web && npm ci)
  else
    (cd web && npm install)
  fi

  log "Installing api dependencies into api/.venv"
  python3 -m venv api/.venv
  api/.venv/bin/python -m pip install --upgrade pip
  api/.venv/bin/pip install -r api/requirements-dev.txt

  log "Pre-pulling CUDA base image: $CUDA_BASE_IMAGE"
  pull_with_retry "$CUDA_BASE_IMAGE" "$PULL_RETRY_COUNT" "$PULL_RETRY_DELAY"

  if [[ "${SKIP_NODE_PREPULL:-0}" != "1" ]]; then
    log "Pre-pulling Node base image: $NODE_BASE_IMAGE"
    pull_with_retry "$NODE_BASE_IMAGE" "$PULL_RETRY_COUNT" "$PULL_RETRY_DELAY"
  else
    log "Skipping Node base image pre-pull (SKIP_NODE_PREPULL=1)"
  fi

  log "Building Docker images"
  CUDA_BASE_IMAGE="$CUDA_BASE_IMAGE" NODE_BASE_IMAGE="$NODE_BASE_IMAGE" "${COMPOSE_CMD[@]}" build

  log "Install complete"
}

main "$@"