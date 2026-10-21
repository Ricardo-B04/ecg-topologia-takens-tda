#!/usr/bin/env bash
set -euo pipefail

if ! command -v kaggle >/dev/null 2>&1; then
  echo 'Instala y configura la CLI de Kaggle antes de ejecutar este script.' >&2
  exit 1
fi

cd "$(dirname "$0")"
mkdir -p data
for archivo in mitbih_train.csv mitbih_test.csv; do
  if [[ ! -f "data/$archivo" ]]; then
    kaggle datasets download shayanfazeli/heartbeat -f "$archivo" -p data --unzip
  fi
done
python3 verificar_datos.py
