#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${1:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-dev}"

cd "$ROOT"

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        "Python 3.10+ is required. Re-run with: "
        "scripts/bootstrap_env.sh /path/to/python3.12"
    )
PY

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/pip" install -e ".[$INSTALL_EXTRAS]"

cat <<EOF
Environment ready:
  $VENV_DIR/bin/python

Try:
  source "$VENV_DIR/bin/activate"
  phantom-twin phase1-summary
  scripts/smoke_test_cli.sh "$VENV_DIR/bin/python"
EOF
