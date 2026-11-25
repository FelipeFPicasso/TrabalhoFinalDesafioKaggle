#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
$PY 01_eda.py
$PY 02_modelos.py
$PY 03_comparacao.py
