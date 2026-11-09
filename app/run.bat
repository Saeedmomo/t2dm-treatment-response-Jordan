@echo off
REM Double-click to launch the LightGBM HbA1c-trajectory checker.
cd /d "%~dp0"
python -m pip install --quiet flask lightgbm numpy
echo Starting server... open http://127.0.0.1:5000 in your browser
python app.py
pause
