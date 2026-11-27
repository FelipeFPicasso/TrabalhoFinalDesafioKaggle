#!/usr/bin/env bash
# Roda o pipeline inteiro, na ordem. Precisa de data/train.csv e data/test.csv.
# Para usar outro Python: PY=python3 ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
$PY 01_eda.py                     # análise exploratória
$PY 02_modelos.py                 # baseline, RL, LightGBM, SVM, MLP e ensemble (submissões 00 a 05)
$PY 02b_tentativas_extras.py      # k-NN, MLP com limiar ajustado e stacking (submissões 06 a 08)
$PY 02c_ablacao_familia.py        # atributo de família: com x sem
$PY 02d_importancia_permutacao.py # importância por permutação em cada modelo
$PY 03_comparacao.py              # tabela e gráficos comparativos
