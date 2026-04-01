#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

env \
  -u ALL_PROXY \
  -u all_proxy \
  -u HTTP_PROXY \
  -u HTTPS_PROXY \
  -u http_proxy \
  -u https_proxy \
  uv run python -m uniter.cli train --config configs/kaitong_west_lane_single_region.yaml "$@"
