@echo off
rem Launches the MatrixFractals GUI. First run creates .venv and installs requirements-gui.txt.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    python -m venv .venv || goto :fail
    .venv\Scripts\python.exe -m pip install --upgrade pip
)
if not exist .venv\.deps-installed (
    .venv\Scripts\python.exe -m pip install -r requirements-gui.txt || goto :fail
    echo ok> .venv\.deps-installed
)
.venv\Scripts\python.exe -m gui.app %*
exit /b %errorlevel%
:fail
echo Setup failed. Is Python 3 on PATH?
pause
exit /b 1
