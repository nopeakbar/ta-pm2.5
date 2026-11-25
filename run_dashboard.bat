@echo off
cd /d "%~dp0"
echo Menjalankan Dashboard Evaluasi PM2.5 DKI Jakarta (Streamlit)...
if exist "D:\Career\TA-Fword\env\Scripts\streamlit.exe" (
    "D:\Career\TA-Fword\env\Scripts\streamlit.exe" run app.py
) else (
    streamlit run app.py
)
pause
