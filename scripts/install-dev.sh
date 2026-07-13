#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Create venv if missing
if [ ! -d ".venv" ]; then
  python3.11 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip
python -m pip install -U pip

# Uninstall package if it exists
python -m pip uninstall -y aircraft-maintenance-nlp || true

# Install pinned dependencies if available
if [ -f "requirements.txt" ]; then
  python -m pip install -r requirements.txt
fi

# Install deps + editable package
python -m pip install -e .