#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-testlm}"
APP_STORAGE_DIR="${APP_STORAGE_DIR:-$ROOT_DIR/storage}"
CHECK_SERVER="${CHECK_SERVER:-0}"
API_URL="${API_URL:-http://localhost:8000}"

log() {
  echo "[check-conda] $*"
}

fail() {
  echo "[check-conda][error] $*" >&2
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
  require_cmd nvidia-smi

  env_exists || fail "Conda environment '$ENV_NAME' does not exist. Run scripts/setup_conda.sh first."

  log "Checking NVIDIA GPU visibility"
  nvidia-smi >/dev/null

  log "Checking Python CUDA and TensorRT runtime in Conda environment '$ENV_NAME'"
  APP_STORAGE_DIR="$APP_STORAGE_DIR" conda run -n "$ENV_NAME" python - <<'PY'
import os
from pathlib import Path

import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is not available")
print(f"device={torch.cuda.get_device_name(0)}")

import tensorrt as trt
from cuda import cudart

print(f"tensorrt={trt.__version__}")
if not hasattr(cudart, "cudaMalloc"):
    raise SystemExit("CUDA Python runtime bindings are incomplete")

storage = Path(os.environ["APP_STORAGE_DIR"]).resolve()
for relative in ("uploads/models", "uploads/datasets", "jobs", "results"):
    path = storage / relative
    path.mkdir(parents=True, exist_ok=True)
    if not os.access(path, os.W_OK):
        raise SystemExit(f"Storage path is not writable: {path}")
print(f"storage={storage}")
PY

  if [[ "$CHECK_SERVER" == "1" ]]; then
    log "Checking running API at $API_URL"
    API_URL="$API_URL" conda run -n "$ENV_NAME" python - <<'PY'
import json
import os
import urllib.request

api_url = os.environ["API_URL"].rstrip("/")
for path in ("/health", "/runtime"):
    with urllib.request.urlopen(f"{api_url}{path}", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    print(f"{path}: {payload}")
PY
  fi

  log "Runtime check passed"
}

main "$@"
