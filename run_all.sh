#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
$PY 01_eda.py
$PY 02_modelos.py
$PY 02b_tentativas_extras.py
$PY 02c_ablacao_familia.py
$PY 02d_importancia_permutacao.py
$PY 03_comparacao.py
