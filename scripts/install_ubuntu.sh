#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

  log "Building Docker images"
  "${COMPOSE_CMD[@]}" build

  log "Install complete"
}

main "$@"