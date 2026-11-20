@echo off
cd /d "%~dp0"
python -c "import sys; assert sys.version_info[:2] == (3, 13), 'Use Python 3.13'"
if errorlevel 1 exit /b 1
if not exist .venv (
    python -m venv .venv
    if errorlevel 1 exit /b 1
)
.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.txt
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m streamlit run app.py
