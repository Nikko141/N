@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   FastText GUI - запуск дипломной демонстрации
echo ============================================================
echo.
echo Если запускается впервые - устанавливаем зависимости:
pip install --quiet streamlit gensim pandas matplotlib numpy
echo.
echo Открываю GUI в браузере...
echo (для остановки - закройте это окно или нажмите Ctrl+C)
echo.
streamlit run gui.py
pause
