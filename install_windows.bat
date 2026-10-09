@echo off
cd /d "%~dp0"
if not exist bertalign_odysseus (
  echo ERREUR : le dossier bertalign_odysseus est introuvable.
  echo Copiez les fichiers de cette V1 a la racine du depot pyOdysseus.
  pause
  exit /b 1
)
py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || py -3.10 -m venv .venv
if errorlevel 1 (
  echo Python 3.10, 3.11 ou 3.12 est requis.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-app.txt
python -m spacy download fr_core_news_sm
if errorlevel 1 (
  echo L'installation a echoue.
  pause
  exit /b 1
)
echo Installation terminee. Lancez launch_windows.bat.
pause
