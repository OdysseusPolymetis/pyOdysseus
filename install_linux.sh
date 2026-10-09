#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-app.txt
python -m spacy download fr_core_news_sm
echo "Installation terminée. Lancez ./launch_linux.sh"
