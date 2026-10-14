@echo off
REM One-time setup (Windows). Run from the project folder.
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo Setup done. Edit .env if you want MySQL, then run run_app.bat
pause
