#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT_DIR/scripts/install_ubuntu.sh"
SKIP_BUILD="${SKIP_BUILD:-1}" \
SKIP_CUDA_PREPULL="${SKIP_CUDA_PREPULL:-1}" \
SKIP_NODE_PREPULL="${SKIP_NODE_PREPULL:-1}" \
exec "$ROOT_DIR/scripts/start_ubuntu_production.sh" "$@"