@echo off
REM ChandraMatch - Windows environment setup
REM Run this from VS Code's terminal (or double-click it) inside your
REM project folder. Safe to re-run if something fails partway through.

echo === Creating virtual environment (.venv) ===
python -m venv .venv
if errorlevel 1 (
    echo Could not find 'python'. Open VS Code, Ctrl+Shift+P -^> "Python: Select Interpreter"
    echo to confirm Python is installed and on PATH, then re-run this script.
    pause
    exit /b 1
)

echo === Activating .venv ===
call .venv\Scripts\activate.bat

echo === Installing packages ===
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ================================================================
echo Done. To use this environment again in a NEW terminal, run:
echo     .venv\Scripts\activate.bat
echo In VS Code: Ctrl+Shift+P -^> "Python: Select Interpreter" -^>
echo pick the one that says .\.venv\Scripts\python.exe
echo ================================================================
pause
