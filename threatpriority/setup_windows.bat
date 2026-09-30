@echo off
cd /d "%~dp0"
py -3.12 --version
if errorlevel 1 goto failed
if not exist .venv\Scripts\python.exe py -3.12 -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Run run_windows.bat and open http://127.0.0.1:5050
pause
exit /b 0
:failed
echo Setup failed. Read the error above and the README troubleshooting section.
pause
exit /b 1
