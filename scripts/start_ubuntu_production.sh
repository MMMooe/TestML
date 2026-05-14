#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04}"
GPU_SMOKE_IMAGE="${GPU_SMOKE_IMAGE:-nvidia/cuda:12.1.1-base-ubuntu22.04}"
PULL_RETRY_COUNT="${PULL_RETRY_COUNT:-5}"
PULL_RETRY_DELAY="${PULL_RETRY_DELAY:-15}"

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

  command -v docker >/dev/null 2>&1 || fail "Missing required command: docker"
  command -v nvidia-smi >/dev/null 2>&1 || fail "Missing required command: nvidia-smi"

  detect_compose

  log "Checking NVIDIA GPU visibility"
  nvidia-smi >/dev/null

  if [[ "${SKIP_GPU_SMOKE_TEST:-0}" != "1" ]]; then
    log "Running Docker GPU smoke test"
    docker run --rm --gpus all --entrypoint /bin/sh "$GPU_SMOKE_IMAGE" -c 'test -e /dev/nvidiactl || test -e /dev/nvidia0' >/dev/null
  else
    log "Skipping Docker GPU smoke test (SKIP_GPU_SMOKE_TEST=1)"
  fi

  if [[ "${SKIP_CUDA_PREPULL:-0}" != "1" ]]; then
    log "Pre-pulling CUDA base image: $CUDA_BASE_IMAGE"
    pull_with_retry "$CUDA_BASE_IMAGE" "$PULL_RETRY_COUNT" "$PULL_RETRY_DELAY"
  else
    log "Skipping CUDA base image pre-pull (SKIP_CUDA_PREPULL=1)"
  fi

  unset DOCKER_HOST DOCKER_CONTEXT
  if docker context inspect default >/dev/null 2>&1; then
    docker context use default >/dev/null
  fi

  log "Starting production stack"
  CUDA_BASE_IMAGE="$CUDA_BASE_IMAGE" "${COMPOSE_CMD[@]}" up --build "$@"
}

main "$@"