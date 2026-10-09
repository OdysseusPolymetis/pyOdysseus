@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo Lancez d'abord install_windows.bat.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m streamlit run app.py
