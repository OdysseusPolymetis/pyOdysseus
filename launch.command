#!/bin/bash
set -e

# macOS/Apple Silicon: évite les crashs natifs FAISS/OpenMP observés au premier alignement.
export OMP_NUM_THREADS=1
export FAISS_OPT_LEVEL=generic

cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  echo "L'application n'est pas encore installée. Lancez d'abord install.command."
  read -r -p "Appuyez sur Entrée pour fermer..."
  exit 1
fi
source .venv/bin/activate
python -m streamlit run app.py --server.maxUploadSize 4096 --server.maxMessageSize 4096
