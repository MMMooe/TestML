#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE:-nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04}"
NODE_BASE_IMAGE="${NODE_BASE_IMAGE:-node:20-alpine}"
GPU_SMOKE_IMAGE="${GPU_SMOKE_IMAGE:-nvidia/cuda:12.1.1-base-ubuntu22.04}"
UBUNTU_APT_MIRROR="${UBUNTU_APT_MIRROR:-http://archive.ubuntu.com/ubuntu}"
UBUNTU_SECURITY_APT_MIRROR="${UBUNTU_SECURITY_APT_MIRROR:-http://security.ubuntu.com/ubuntu}"
NVIDIA_APT_MIRROR="${NVIDIA_APT_MIRROR:-https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64}"
APT_RETRIES="${APT_RETRIES:-3}"
APT_TIMEOUT="${APT_TIMEOUT:-20}"
APT_FORCE_IPV4="${APT_FORCE_IPV4:-true}"
DOCKER_BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-host}"
PULL_RETRY_COUNT="${PULL_RETRY_COUNT:-10}"
PULL_RETRY_DELAY="${PULL_RETRY_DELAY:-20}"
DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}"
COMPOSE_DOCKER_CLI_BUILD="${COMPOSE_DOCKER_CLI_BUILD:-0}"

log() {
  echo "[start] $*"
}

warn() {
  echo "[start][warn] $*" >&2
}

fail() {
  echo "[start][error] $*" >&2
  exit 1
}

run_step() {
  local description="$1"
  shift

  log "$description"
  if "$@"; then
    log "PASS: $description"
    return 0
  fi

  local status=$?
  echo "[start][error] FAIL: $description (exit $status)" >&2
  return "$status"
}

show_compose_status() {
  log "Current Compose service status"
  "${COMPOSE_CMD[@]}" ps || warn "Could not read Compose service status"
}

detect_compose() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
    log "Using Docker Compose command: docker compose"
    return
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
    log "Using Docker Compose command: docker-compose"
    return
  fi
  fail "Docker Compose not found. Install docker compose v2 or docker-compose."
}

use_local_docker_context() {
  unset DOCKER_HOST DOCKER_CONTEXT
  if docker context inspect default >/dev/null 2>&1; then
    docker context use default >/dev/null || fail "Failed to switch Docker context to default"
    log "Using Docker context: default"
  else
    warn "Docker context 'default' was not found; using current Docker daemon settings"
  fi
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

  log "Repository root: $ROOT_DIR"
  log "Docker version: $(docker --version)"

  use_local_docker_context
  detect_compose

  log "Configuration: CUDA_BASE_IMAGE=$CUDA_BASE_IMAGE NODE_BASE_IMAGE=$NODE_BASE_IMAGE GPU_SMOKE_IMAGE=$GPU_SMOKE_IMAGE"
  log "Configuration: SKIP_BUILD=${SKIP_BUILD:-0} SKIP_GPU_SMOKE_TEST=${SKIP_GPU_SMOKE_TEST:-0} SKIP_CUDA_PREPULL=${SKIP_CUDA_PREPULL:-0} SKIP_NODE_PREPULL=${SKIP_NODE_PREPULL:-0}"

  run_step "Checking NVIDIA GPU visibility with nvidia-smi" nvidia-smi

  if [[ "${SKIP_GPU_SMOKE_TEST:-0}" != "1" ]]; then
    log "Pre-pulling GPU smoke-test image: $GPU_SMOKE_IMAGE"
    pull_with_retry "$GPU_SMOKE_IMAGE" "$PULL_RETRY_COUNT" "$PULL_RETRY_DELAY"
    log "Running Docker GPU smoke test"
    if docker run --rm --pull never --gpus all --entrypoint /bin/sh "$GPU_SMOKE_IMAGE" -c 'echo "NVIDIA devices inside container:"; ls -l /dev/nvidia* 2>&1; test -e /dev/nvidiactl || test -e /dev/nvidia0'; then
      log "PASS: Docker GPU smoke test"
    else
      cat >&2 <<'EOF'
[start][error] FAIL: Docker GPU smoke test
[start][error] Docker can run, and the host nvidia-smi works, but GPU devices were not visible inside the test container.
[start][error] Install or reconfigure NVIDIA Container Toolkit, then restart Docker:
[start][error]   sudo apt-get install -y nvidia-container-toolkit
[start][error]   sudo nvidia-ctk runtime configure --runtime=docker
[start][error]   sudo systemctl restart docker
[start][error] To bypass only this pre-check temporarily:
[start][error]   SKIP_GPU_SMOKE_TEST=1 SKIP_BUILD=1 SKIP_CUDA_PREPULL=1 SKIP_NODE_PREPULL=1 ./scripts/start_ubuntu_production.sh
EOF
      exit 1
    fi
  else
    log "Skipping Docker GPU smoke test (SKIP_GPU_SMOKE_TEST=1)"
  fi

  if [[ "${SKIP_CUDA_PREPULL:-0}" != "1" ]]; then
    log "Pre-pulling CUDA base image: $CUDA_BASE_IMAGE"
    pull_with_retry "$CUDA_BASE_IMAGE" "$PULL_RETRY_COUNT" "$PULL_RETRY_DELAY"
  else
    log "Skipping CUDA base image pre-pull (SKIP_CUDA_PREPULL=1)"
  fi

  if [[ "${SKIP_NODE_PREPULL:-0}" != "1" ]]; then
    log "Pre-pulling Node base image: $NODE_BASE_IMAGE"
    pull_with_retry "$NODE_BASE_IMAGE" "$PULL_RETRY_COUNT" "$PULL_RETRY_DELAY"
  else
    log "Skipping Node base image pre-pull (SKIP_NODE_PREPULL=1)"
  fi

  if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    log "Building Docker images"
    log "Using Docker build network: $DOCKER_BUILD_NETWORK"
    log "Using APT mirrors: ubuntu=$UBUNTU_APT_MIRROR security=$UBUNTU_SECURITY_APT_MIRROR nvidia=$NVIDIA_APT_MIRROR"
    CUDA_BASE_IMAGE="$CUDA_BASE_IMAGE" NODE_BASE_IMAGE="$NODE_BASE_IMAGE" UBUNTU_APT_MIRROR="$UBUNTU_APT_MIRROR" UBUNTU_SECURITY_APT_MIRROR="$UBUNTU_SECURITY_APT_MIRROR" NVIDIA_APT_MIRROR="$NVIDIA_APT_MIRROR" APT_RETRIES="$APT_RETRIES" APT_TIMEOUT="$APT_TIMEOUT" APT_FORCE_IPV4="$APT_FORCE_IPV4" DOCKER_BUILD_NETWORK="$DOCKER_BUILD_NETWORK" DOCKER_BUILDKIT="$DOCKER_BUILDKIT" COMPOSE_DOCKER_CLI_BUILD="$COMPOSE_DOCKER_CLI_BUILD" "${COMPOSE_CMD[@]}" build || fail "Docker image build failed"
    log "PASS: Docker image build"
  else
    log "Skipping Docker image build (SKIP_BUILD=1)"
  fi

  log "Starting production stack"
  if CUDA_BASE_IMAGE="$CUDA_BASE_IMAGE" NODE_BASE_IMAGE="$NODE_BASE_IMAGE" UBUNTU_APT_MIRROR="$UBUNTU_APT_MIRROR" UBUNTU_SECURITY_APT_MIRROR="$UBUNTU_SECURITY_APT_MIRROR" NVIDIA_APT_MIRROR="$NVIDIA_APT_MIRROR" APT_RETRIES="$APT_RETRIES" APT_TIMEOUT="$APT_TIMEOUT" APT_FORCE_IPV4="$APT_FORCE_IPV4" DOCKER_BUILD_NETWORK="$DOCKER_BUILD_NETWORK" "${COMPOSE_CMD[@]}" up --no-build "$@"; then
    log "PASS: Production stack exited cleanly"
  else
    status=$?
    show_compose_status
    fail "Production stack failed or exited unexpectedly (exit $status)"
  fi
}

main "$@"