@echo off
echo ========================================
echo    TransparAI Dashboard - SFLC.in
echo ========================================
echo.
echo Starting TransparAI Procurement Dashboard...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Starting Streamlit dashboard...
streamlit run app.py

pause
