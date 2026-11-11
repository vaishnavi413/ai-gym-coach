@echo off
title AI Gym Coach
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)
cd "Main App"
streamlit run main.py
pause
