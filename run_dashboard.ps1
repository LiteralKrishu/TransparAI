$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
python -c "import sys; assert sys.version_info[:2] == (3, 13), 'Use Python 3.13'"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not (Test-Path '.venv')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& .\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m streamlit run app.py
exit $LASTEXITCODE
