#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PY:-python3}" "$HERE/build_corrected_figures.py" "$@"
