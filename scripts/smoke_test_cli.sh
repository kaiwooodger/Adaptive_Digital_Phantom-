#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${1:-python3}"

cd "$ROOT"

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required for phantom-digital-twin.")
PY

"$PYTHON_BIN" -m phantom_twin.cli phase1-summary >/tmp/phantom_twin_phase1_summary.txt
"$PYTHON_BIN" -m phantom_twin.cli materials-check >/tmp/phantom_twin_materials_check.txt
"$PYTHON_BIN" -m phantom_twin.cli datasets-list >/tmp/phantom_twin_datasets_list.txt

grep -qi "phase" /tmp/phantom_twin_phase1_summary.txt
grep -qi "material" /tmp/phantom_twin_materials_check.txt
grep -qi "dataset" /tmp/phantom_twin_datasets_list.txt

echo "CLI smoke tests passed."
