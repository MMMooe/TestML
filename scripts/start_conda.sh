#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-testlm}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-3000}"
WEB_MODE="${WEB_MODE:-dev}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

origin_host_for_browser() {
  local host="$1"
  if [[ "$host" == "0.0.0.0" || "$host" == "::" ]]; then
    echo "localhost"
    return
  fi
  echo "$host"
}

API_PUBLIC_HOST="${API_PUBLIC_HOST:-$(origin_host_for_browser "$API_HOST")}"
WEB_PUBLIC_HOST="${WEB_PUBLIC_HOST:-$(origin_host_for_browser "$WEB_HOST")}"

APP_MODE="${APP_MODE:-production-cuda}"
APP_REQUIRE_TENSORRT="${APP_REQUIRE_TENSORRT:-true}"
APP_STORAGE_DIR="${APP_STORAGE_DIR:-$ROOT_DIR/storage}"
APP_CORS_ORIGINS="${APP_CORS_ORIGINS:-http://$WEB_PUBLIC_HOST:$WEB_PORT,http://localhost:$WEB_PORT,http://127.0.0.1:$WEB_PORT}"
APP_MAX_UPLOAD_MB="${APP_MAX_UPLOAD_MB:-4096}"
NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-http://$API_PUBLIC_HOST:$API_PORT}"

API_PID=""
WEB_PID=""

log() {
  echo "[start-conda] $*"
}

fail() {
  echo "[start-conda][error] $*" >&2
  exit 1
}

require_cmd() {
  local name="$1"
  command -v "$name" >/dev/null 2>&1 || fail "Missing required command: $name"
}

env_exists() {
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fx "$ENV_NAME" >/dev/null 2>&1
}

cleanup() {
  trap - INT TERM EXIT
  if [[ -n "$WEB_PID" ]] && kill -0 "$WEB_PID" >/dev/null 2>&1; then
    kill "$WEB_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$API_PID" ]] && kill -0 "$API_PID" >/dev/null 2>&1; then
    kill "$API_PID" >/dev/null 2>&1 || true
  fi
  wait "$WEB_PID" "$API_PID" >/dev/null 2>&1 || true
}

wait_for_api() {
  local url="http://localhost:$API_PORT/health"
  local start_time
  start_time="$(date +%s)"

  while true; do
    if API_URL="$url" conda run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import os
import urllib.request

urllib.request.urlopen(os.environ["API_URL"], timeout=1).read()
PY
    then
      return 0
    fi

    if (( $(date +%s) - start_time >= WAIT_TIMEOUT )); then
      return 1
    fi
    sleep 1
  done
}

main() {
  cd "$ROOT_DIR"

  require_cmd conda
  env_exists || fail "Conda environment '$ENV_NAME' does not exist. Run scripts/setup_conda.sh first."

  mkdir -p "$APP_STORAGE_DIR/uploads/models" "$APP_STORAGE_DIR/uploads/datasets" "$APP_STORAGE_DIR/jobs" "$APP_STORAGE_DIR/results"

  trap cleanup INT TERM EXIT

  log "Starting API on http://localhost:$API_PORT"
  (
    cd "$ROOT_DIR/api"
    APP_MODE="$APP_MODE" \
    APP_REQUIRE_TENSORRT="$APP_REQUIRE_TENSORRT" \
    APP_STORAGE_DIR="$APP_STORAGE_DIR" \
    APP_CORS_ORIGINS="$APP_CORS_ORIGINS" \
    APP_MAX_UPLOAD_MB="$APP_MAX_UPLOAD_MB" \
    conda run --no-capture-output -n "$ENV_NAME" python -m uvicorn app.main:app --host "$API_HOST" --port "$API_PORT"
  ) &
  API_PID="$!"

  log "Waiting for API health check"
  if ! wait_for_api; then
    fail "API did not become healthy within ${WAIT_TIMEOUT}s. Check CUDA/TensorRT runtime output above."
  fi

  log "Starting web app on http://localhost:$WEB_PORT (WEB_MODE=$WEB_MODE)"
  if [[ "$WEB_MODE" == "production" ]]; then
    (
      cd "$ROOT_DIR/web"
      NEXT_PUBLIC_API_URL="$NEXT_PUBLIC_API_URL" conda run --no-capture-output -n "$ENV_NAME" npm run build
      NEXT_PUBLIC_API_URL="$NEXT_PUBLIC_API_URL" PORT="$WEB_PORT" HOSTNAME="$WEB_HOST" conda run --no-capture-output -n "$ENV_NAME" npm start -- --hostname "$WEB_HOST" --port "$WEB_PORT"
    ) &
  else
    (
      cd "$ROOT_DIR/web"
      NEXT_PUBLIC_API_URL="$NEXT_PUBLIC_API_URL" conda run --no-capture-output -n "$ENV_NAME" npm run dev -- --hostname "$WEB_HOST" --port "$WEB_PORT"
    ) &
  fi
  WEB_PID="$!"

  log "UI: http://localhost:$WEB_PORT"
  log "API docs: http://localhost:$API_PORT/docs"
  log "Press Ctrl-C to stop both processes"

  set +e
  wait -n "$API_PID" "$WEB_PID"
  status="$?"
  set -e
  exit "$status"
}

main "$@"
