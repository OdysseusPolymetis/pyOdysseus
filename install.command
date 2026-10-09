#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d "bertalign_odysseus" ]; then
  echo "ERREUR : le dossier bertalign_odysseus est introuvable."
  echo "Copiez les fichiers de cette V1 à la racine du dépôt pyOdysseus."
  read -r -p "Appuyez sur Entrée pour fermer..."
  exit 1
fi

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "Python 3 est introuvable. Installez Python 3.10, 3.11 ou 3.12 puis relancez ce fichier."
  read -r -p "Appuyez sur Entrée pour fermer..."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3,10) or sys.version_info >= (3,13):
    raise SystemExit("pyOdysseus V1 attend Python 3.10, 3.11 ou 3.12.")
print("Python", sys.version.split()[0])
PY

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-app.txt
python -m spacy download fr_core_news_sm

echo
echo "Installation terminée."
echo "Vous pouvez maintenant double-cliquer sur launch.command."
read -r -p "Appuyez sur Entrée pour fermer..."
