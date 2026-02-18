@echo off
REM Run the Streamlit app using the workspace virtual environment.
cd /d "%~dp0"
"C:\MachineLearning\venv\Scripts\python.exe" -m streamlit run app.py
pause
