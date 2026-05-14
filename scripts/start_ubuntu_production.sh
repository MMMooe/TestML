#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
  echo "[start] $*"
}

fail() {
  echo "[start][error] $*" >&2
  exit 1
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

main() {
  cd "$ROOT_DIR"

  command -v docker >/dev/null 2>&1 || fail "Missing required command: docker"
  command -v nvidia-smi >/dev/null 2>&1 || fail "Missing required command: nvidia-smi"

  detect_compose

  log "Checking NVIDIA GPU visibility"
  nvidia-smi >/dev/null

  if [[ "${SKIP_GPU_SMOKE_TEST:-0}" != "1" ]]; then
    log "Running Docker GPU smoke test"
    docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi >/dev/null
  else
    log "Skipping Docker GPU smoke test (SKIP_GPU_SMOKE_TEST=1)"
  fi

  unset DOCKER_HOST DOCKER_CONTEXT
  if docker context inspect default >/dev/null 2>&1; then
    docker context use default >/dev/null
  fi

  log "Starting production stack"
  "${COMPOSE_CMD[@]}" up --build "$@"
}

main "$@"