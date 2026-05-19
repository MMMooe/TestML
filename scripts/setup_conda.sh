#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-testlm}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/environment.yml}"
INSTALL_TENSORRT="${INSTALL_TENSORRT:-0}"
TENSORRT_PACKAGE="${TENSORRT_PACKAGE:-tensorrt}"
SKIP_RUNTIME_CHECK="${SKIP_RUNTIME_CHECK:-0}"

log() {
  echo "[setup-conda] $*"
}

fail() {
  echo "[setup-conda][error] $*" >&2
  exit 1
}

require_cmd() {
  local name="$1"
  command -v "$name" >/dev/null 2>&1 || fail "Missing required command: $name"
}

env_exists() {
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fx "$ENV_NAME" >/dev/null 2>&1
}

main() {
  cd "$ROOT_DIR"

  require_cmd conda
  [[ -f "$ENV_FILE" ]] || fail "Environment file not found: $ENV_FILE"

  if env_exists; then
    log "Updating Conda environment '$ENV_NAME' from $ENV_FILE"
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
  else
    log "Creating Conda environment '$ENV_NAME' from $ENV_FILE"
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"
  fi

  log "Ensuring API GPU dependencies are installed"
  (cd "$ROOT_DIR/api" && conda run -n "$ENV_NAME" python -m pip install -r requirements-gpu.txt)

  if [[ "$INSTALL_TENSORRT" == "1" ]]; then
    log "Installing TensorRT Python package '$TENSORRT_PACKAGE' from NVIDIA PyPI"
    conda run -n "$ENV_NAME" python -m pip install --extra-index-url https://pypi.nvidia.com "$TENSORRT_PACKAGE"
  else
    log "Skipping TensorRT package install. Set INSTALL_TENSORRT=1 to attempt pip installation from NVIDIA PyPI."
  fi

  log "Installing web dependencies"
  if [[ -f "$ROOT_DIR/web/package-lock.json" ]]; then
    (cd "$ROOT_DIR/web" && conda run -n "$ENV_NAME" npm ci)
  else
    (cd "$ROOT_DIR/web" && conda run -n "$ENV_NAME" npm install)
  fi

  log "Creating runtime storage directories"
  mkdir -p "$ROOT_DIR/storage/uploads/models" "$ROOT_DIR/storage/uploads/datasets" "$ROOT_DIR/storage/jobs" "$ROOT_DIR/storage/results"

  if [[ "$SKIP_RUNTIME_CHECK" == "1" ]]; then
    log "Skipping runtime check (SKIP_RUNTIME_CHECK=1)"
  else
    ENV_NAME="$ENV_NAME" APP_STORAGE_DIR="$ROOT_DIR/storage" "$ROOT_DIR/scripts/check_conda_runtime.sh"
  fi

  log "Conda setup complete"
  log "Start the app with: ENV_NAME=$ENV_NAME ./scripts/start_conda.sh"
}

main "$@"
