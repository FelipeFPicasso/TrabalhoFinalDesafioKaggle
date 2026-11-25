#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
KAGGLE=${KAGGLE:-.venv/bin/kaggle}
arquivos=("${@:-}")
[ -z "${arquivos[0]}" ] && arquivos=($(ls submissoes/))
for f in "${arquivos[@]}"; do
  echo ">> $f"
  $KAGGLE competitions submit -c spaceship-titanic -f "submissoes/$f" -m "$f"
done
$KAGGLE competitions submissions -c spaceship-titanic
